#!/usr/bin/env python3
"""Naive PDF RAG over files in pdf_files/.

This version is intentionally small:
- extract text from PDFs
- split pages into overlapping chunks
- rank chunks with BM25
- produce an extractive answer with source citations
"""


from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PDF_DIR = Path("pdf_files") #read pdf files from
DEFAULT_INDEX_PATH = Path("rag_index.json") #write index to this file
DEFAULT_OPENAI_MODEL = "gpt-5.5"
TOKEN_RE = re.compile(r"(?u)\b\w+\b") #tokenize text
SENTENCE_RE = re.compile(r"(?<=[.!?。！？;:])\s+|\n+") #split text into sentences


def find_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable

    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin" / name
    if bundled.exists():
        return str(bundled)

    homebrew = Path("/opt/homebrew/bin") / name
    if homebrew.exists():
        return str(homebrew)

    return None


@dataclass
class Chunk:
    id: int
    source: str
    document: str
    page: int
    chunk_on_page: int
    text: str
    term_counts: dict[str, int]
    length: int


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]


def chunk_words(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return

    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            yield chunk
        if start + chunk_size >= len(words):
            break


def extract_with_pdftotext(pdf_path: Path, timeout: int) -> list[tuple[int, str]]:
    executable = find_executable("pdftotext")
    if not executable:
        raise FileNotFoundError("pdftotext was not found on PATH")

    completed = subprocess.run(
        [executable, "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    raw_pages = completed.stdout.split("\f")
    pages = [(index, clean_text(text)) for index, text in enumerate(raw_pages, start=1)]
    if pages and not pages[-1][1]:
        pages.pop()
    return pages


def page_number_from_image(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    if match:
        return int(match.group(1))
    return 1


def extract_with_ocr(pdf_path: Path, timeout: int, dpi: int, lang: str) -> list[tuple[int, str]]:
    pdftoppm = find_executable("pdftoppm")
    tesseract = find_executable("tesseract")
    if not pdftoppm:
        raise FileNotFoundError("pdftoppm was not found")
    if not tesseract:
        raise FileNotFoundError("tesseract was not found")

    with tempfile.TemporaryDirectory(prefix="naive_rag_ocr_") as temp_dir:
        prefix = str(Path(temp_dir) / "page")
        subprocess.run(
            [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), prefix],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        image_paths = sorted(Path(temp_dir).glob("page-*.png"), key=page_number_from_image)
        pages: list[tuple[int, str]] = []
        for image_path in image_paths:
            completed = subprocess.run(
                [tesseract, str(image_path), "stdout", "-l", lang, "--psm", "6"],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            pages.append((page_number_from_image(image_path), clean_text(completed.stdout)))
    return pages


def extract_with_pypdfium2(pdf_path: Path) -> list[tuple[int, str]]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    pages: list[tuple[int, str]] = []
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_range() or ""
            finally:
                textpage.close()
                page.close()
            pages.append((index + 1, clean_text(text)))
    finally:
        pdf.close()
    return pages


def extract_with_pypdf(pdf_path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path), strict=False)
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # Keep building even if one page is odd.
            print(f"  warning: page {index} failed pypdf extraction: {exc}", file=sys.stderr)
            text = ""
        pages.append((index, clean_text(text)))
    return pages


def extract_with_pdfplumber(pdf_path: Path) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                print(
                    f"  warning: page {index} failed pdfplumber extraction: {exc}",
                    file=sys.stderr,
                )
                text = ""
            pages.append((index, clean_text(text)))
    return pages


def _python_extract_worker(extractor: str, pdf_path: str, queue: mp.Queue) -> None:
    try:
        path = Path(pdf_path)
        if extractor == "pypdfium2":
            pages = extract_with_pypdfium2(path)
        elif extractor == "pdfplumber":
            pages = extract_with_pdfplumber(path)
        elif extractor == "pypdf":
            pages = extract_with_pypdf(path)
        else:
            raise ValueError(f"Unknown Python extractor: {extractor}")
        queue.put({"pages": pages})
    except BaseException as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def extract_with_python_timeout(extractor: str, pdf_path: Path, timeout: int) -> list[tuple[int, str]]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_python_extract_worker, args=(extractor, str(pdf_path), queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(2)
        raise TimeoutError(f"{extractor} timed out after {timeout}s")

    if queue.empty():
        raise RuntimeError(f"{extractor} exited with code {process.exitcode} without returning data")

    payload = queue.get()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["pages"]


def extract_pdf_pages(
    pdf_path: Path,
    extractor: str,
    timeout: int,
    ocr_timeout: int,
    ocr_dpi: int,
    ocr_lang: str,
) -> list[tuple[int, str]]:
    extractors = [extractor]
    if extractor == "auto":
        extractors = ["pdftotext", "pypdfium2", "pdfplumber", "pypdf", "ocr"]

    pages: list[tuple[int, str]] = []
    errors: list[str] = []

    for name in extractors:
        try:
            if name == "pdftotext":
                pages = extract_with_pdftotext(pdf_path, timeout)
            elif name == "ocr":
                pages = extract_with_ocr(pdf_path, ocr_timeout, ocr_dpi, ocr_lang)
            elif name in {"pypdfium2", "pdfplumber", "pypdf"}:
                pages = extract_with_python_timeout(name, pdf_path, timeout)
            else:
                raise ValueError(f"Unknown extractor: {name}")
        except ImportError as exc:
            errors.append(f"{name}: missing dependency {exc.name}")
            continue
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{name}: {exc}")
            continue
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue

        if sum(len(text) for _, text in pages) > 0:
            return pages

        errors.append(f"{name}: extracted no text")

    if errors:
        print(f"  warning: extraction issues for {pdf_path.name}: {'; '.join(errors)}", file=sys.stderr)
    return pages


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {pdf_dir}")
    return pdfs


def build_index(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.pdf_dir)
    pdfs = discover_pdfs(pdf_dir)
    chunks: list[Chunk] = []
    document_stats: list[dict[str, int | str]] = []
    skipped_pages = 0

    for pdf_path in pdfs:
        print(f"Reading {pdf_path.name} ...", flush=True)
        pages = extract_pdf_pages(
            pdf_path,
            args.extractor,
            args.extract_timeout,
            args.ocr_timeout,
            args.ocr_dpi,
            args.ocr_lang,
        )
        doc_chunks = 0
        doc_text_pages = 0

        for page_number, page_text in pages:
            if len(page_text) < args.min_page_chars:
                skipped_pages += 1
                continue

            doc_text_pages += 1
            for chunk_on_page, text in enumerate(
                chunk_words(page_text, args.chunk_size, args.overlap), start=1
            ):
                terms = Counter(tokenize(text))
                if not terms:
                    continue

                chunks.append(
                    Chunk(
                        id=len(chunks),
                        source=str(pdf_path),
                        document=pdf_path.name,
                        page=page_number,
                        chunk_on_page=chunk_on_page,
                        text=text,
                        term_counts=dict(terms),
                        length=sum(terms.values()),
                    )
                )
                doc_chunks += 1

        document_stats.append(
            {
                "document": pdf_path.name,
                "pages": len(pages),
                "text_pages": doc_text_pages,
                "chunks": doc_chunks,
            }
        )
        print(f"  pages={len(pages)} text_pages={doc_text_pages} chunks={doc_chunks}", flush=True)

    if not chunks:
        raise SystemExit(
            "No text chunks were created. These PDFs may be scanned images; run OCR first, then rebuild."
        )

    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(chunk.term_counts.keys())

    avgdl = sum(chunk.length for chunk in chunks) / len(chunks)
    index = {
        "version": 1,
        "settings": {
            "pdf_dir": str(pdf_dir),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "min_page_chars": args.min_page_chars,
        },
        "stats": {
            "documents": len(pdfs),
            "chunks": len(chunks),
            "avg_chunk_terms": avgdl,
            "skipped_pages": skipped_pages,
        },
        "document_stats": document_stats,
        "doc_freq": dict(doc_freq),
        "chunks": [asdict(chunk) for chunk in chunks],
    }

    output_path = Path(args.index_path)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(chunks)} chunks to {output_path}")


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}. Build it first with: python3 naive_rag.py build")
    return json.loads(index_path.read_text(encoding="utf-8"))


def bm25_search(index: dict, query: str, top_k: int) -> list[tuple[float, dict]]:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return []

    chunks = index["chunks"]
    doc_freq = index["doc_freq"]
    total_chunks = len(chunks)
    avgdl = float(index["stats"]["avg_chunk_terms"])
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, dict]] = []

    for chunk in chunks:
        term_counts = chunk["term_counts"]
        chunk_len = max(1, int(chunk["length"]))
        score = 0.0

        for term, query_count in query_terms.items():
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = int(doc_freq.get(term, 0))
            idf = math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * chunk_len / avgdl)
            score += query_count * idf * (tf * (k1 + 1)) / denom

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def best_sentences(query: str, results: list[tuple[float, dict]], max_sentences: int) -> list[str]:
    query_terms = set(tokenize(query))
    selected: list[str] = []
    seen: set[str] = set()

    for _, chunk in results:
        sentences = [clean_text(part) for part in SENTENCE_RE.split(chunk["text"]) if clean_text(part)]
        if not sentences:
            sentences = [chunk["text"]]

        ranked = []
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            overlap = len(query_terms & sentence_terms)
            ranked.append((overlap, len(sentence), sentence))

        for overlap, _, sentence in sorted(ranked, key=lambda item: (-item[0], item[1])):
            if overlap == 0 and selected:
                continue
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(sentence)
            if len(selected) >= max_sentences:
                return selected

    return selected


def format_citation(chunk: dict) -> str:
    return f"{chunk['document']} p.{chunk['page']} chunk {chunk['chunk_on_page']}"


def format_context(results: list[tuple[float, dict]], context_chars: int) -> str:
    context_blocks = []
    for rank, (_, chunk) in enumerate(results, start=1):
        citation = format_citation(chunk)
        text = clean_text(chunk["text"])[:context_chars]
        context_blocks.append(f"[{rank}] {citation}\n{text}")
    return "\n\n".join(context_blocks)


def extract_response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()

    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def call_openai_llm(
    question: str,
    results: list[tuple[float, dict]],
    model: str,
    max_output_tokens: int,
    context_chars: int,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    context = format_context(results, context_chars)
    prompt = f"""You are a Vietnamese legal/regulatory RAG assistant.

Answer the user's question using only the context below.
Write in Vietnamese.
Cite sources inline using bracket numbers like [1] or [2].
If the context does not contain enough information, say: "Không tìm thấy đủ thông tin trong tài liệu được cung cấp."
Do not invent facts outside the context.

Question:
{question}

Context:
{context}
"""

    body = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    text = extract_response_text(payload)
    if not text:
        raise RuntimeError("OpenAI API returned no text output.")
    return text


def ask_index(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index_path))
    results = bm25_search(index, args.question, args.top_k)
    if not results:
        print("No matching chunks found.")
        return

    if args.llm == "openai":
        print("\nLLM answer")
        print("----------")
        try:
            print(
                call_openai_llm(
                    args.question,
                    results,
                    args.llm_model,
                    args.llm_max_output_tokens,
                    args.llm_context_chars,
                )
            )
        except RuntimeError as exc:
            print(f"LLM error: {exc}")
            print("\nFalling back to extractive answer.")
            print("-------------------------------")
            sentences = best_sentences(args.question, results, args.max_sentences)
            if sentences:
                for sentence in sentences:
                    print(f"- {sentence}")
            else:
                print(results[0][1]["text"][: args.preview_chars].strip())
    else:
        print("\nExtractive answer")
        print("-----------------")
        sentences = best_sentences(args.question, results, args.max_sentences)
        if sentences:
            for sentence in sentences:
                print(f"- {sentence}")
        else:
            print(results[0][1]["text"][: args.preview_chars].strip())

    print("\nSources")
    print("-------")
    for rank, (score, chunk) in enumerate(results, start=1):
        preview = clean_text(chunk["text"])[: args.preview_chars]
        print(f"[{rank}] score={score:.3f} {format_citation(chunk)}")
        print(f"    {preview}")


def inspect_index(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index_path))
    stats = index["stats"]
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Average chunk terms: {stats['avg_chunk_terms']:.1f}")
    print(f"Skipped low-text pages: {stats['skipped_pages']}")
    print("\nPer document")
    for item in index["document_stats"]:
        print(
            f"- {item['document']}: pages={item['pages']} "
            f"text_pages={item['text_pages']} chunks={item['chunks']}"
        )


def add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Folder containing PDF files.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Where to write the JSON index.")
    parser.add_argument("--chunk-size", type=int, default=450, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap in words.")
    parser.add_argument(
        "--min-page-chars",
        type=int,
        default=40,
        help="Skip pages with fewer extracted characters than this value.",
    )
    parser.add_argument(
        "--extractor",
        choices=["auto", "pdftotext", "pypdfium2", "pypdf", "pdfplumber", "ocr"],
        default="auto",
        help="PDF text extractor to use.",
    )
    parser.add_argument(
        "--extract-timeout",
        type=int,
        default=10,
        help="Timeout in seconds per extraction attempt.",
    )
    parser.add_argument("--ocr-lang", default="eng", help="Tesseract language code, for example eng or vie+eng.")
    parser.add_argument("--ocr-dpi", type=int, default=120, help="DPI used when rendering PDFs for OCR.")
    parser.add_argument("--ocr-timeout", type=int, default=180, help="Timeout in seconds for OCR rendering/OCR.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive PDF RAG over a local pdf_files folder.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Extract PDFs and build rag_index.json.")
    add_common_build_args(build_parser)
    build_parser.set_defaults(func=build_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the built index.")
    ask_parser.add_argument("question", help="Question to search for.")
    ask_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to rag_index.json.")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    ask_parser.add_argument("--max-sentences", type=int, default=5, help="Number of answer sentences.")
    ask_parser.add_argument("--preview-chars", type=int, default=420, help="Characters shown per source.")
    ask_parser.add_argument("--llm", choices=["none", "openai"], default="none", help="Use an LLM to write the answer.")
    ask_parser.add_argument(
        "--llm-model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        help="OpenAI model to use when --llm openai is set.",
    )
    ask_parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=800,
        help="Maximum output tokens for the LLM answer.",
    )
    ask_parser.add_argument(
        "--llm-context-chars",
        type=int,
        default=2500,
        help="Maximum characters included from each retrieved chunk.",
    )
    ask_parser.set_defaults(func=ask_index)

    inspect_parser = subparsers.add_parser("inspect", help="Show index statistics.")
    inspect_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to rag_index.json.")
    inspect_parser.set_defaults(func=inspect_index)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
