#!/usr/bin/env python3
"""FAISS + OpenAI RAG over PDFs in pdf_files/.

This version is intentionally separate from naive_rag.py and advanced_rag.py.
It creates a FAISS vector index on disk and keeps chunk metadata next to it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


DEFAULT_PDF_DIR = Path("pdf_files")
DEFAULT_INDEX_DIR = Path("faiss_index")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-5.5"
SENTENCE_RE = re.compile(r"(?<=[.!?。！？;:])\s+|\n+")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


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


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {pdf_dir}")
    return pdfs


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

    with tempfile.TemporaryDirectory(prefix="faiss_rag_ocr_") as temp_dir:
        prefix = str(Path(temp_dir) / "page")
        subprocess.run(
            [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), prefix],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        pages: list[tuple[int, str]] = []
        image_paths = sorted(Path(temp_dir).glob("page-*.png"), key=page_number_from_image)
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


def extract_with_pdftotext(pdf_path: Path, timeout: int) -> list[tuple[int, str]]:
    executable = find_executable("pdftotext")
    if not executable:
        raise FileNotFoundError("pdftotext was not found")

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


def extract_pdf_pages(args: argparse.Namespace, pdf_path: Path) -> list[tuple[int, str]]:
    if args.extractor == "ocr":
        return extract_with_ocr(pdf_path, args.ocr_timeout, args.ocr_dpi, args.ocr_lang)
    if args.extractor == "pdftotext":
        return extract_with_pdftotext(pdf_path, args.extract_timeout)

    errors: list[str] = []
    for extractor in ("pdftotext", "ocr"):
        try:
            if extractor == "pdftotext":
                pages = extract_with_pdftotext(pdf_path, args.extract_timeout)
            else:
                pages = extract_with_ocr(pdf_path, args.ocr_timeout, args.ocr_dpi, args.ocr_lang)
        except Exception as exc:
            errors.append(f"{extractor}: {exc}")
            continue

        if sum(len(text) for _, text in pages) > 0:
            return pages
        errors.append(f"{extractor}: extracted no text")

    print(f"  warning: extraction issues for {pdf_path.name}: {'; '.join(errors)}")
    return []


def openai_request(path: str, body: dict, timeout: int = 120) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    request = urllib.request.Request(
        f"https://api.openai.com/v1/{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc


def embed_texts(texts: list[str], model: str, batch_size: int, retries: int = 3) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(retries):
            try:
                payload = openai_request("embeddings", {"model": model, "input": batch})
                break
            except RuntimeError:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        vectors.extend(item["embedding"] for item in payload["data"])
        print(f"  embedded {min(start + len(batch), len(texts))}/{len(texts)} chunks", flush=True)

    matrix = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_chunks(args: argparse.Namespace) -> list[dict]:
    chunks: list[dict] = []
    for pdf_path in discover_pdfs(Path(args.pdf_dir)):
        print(f"Reading {pdf_path.name} ...", flush=True)
        pages = extract_pdf_pages(args, pdf_path)
        doc_chunks = 0
        text_pages = 0
        for page_number, page_text in pages:
            if len(page_text) < args.min_page_chars:
                continue
            text_pages += 1
            for chunk_on_page, text in enumerate(chunk_words(page_text, args.chunk_size, args.overlap), start=1):
                chunks.append(
                    {
                        "id": len(chunks),
                        "source": str(pdf_path),
                        "document": pdf_path.name,
                        "page": page_number,
                        "chunk_on_page": chunk_on_page,
                        "text": text,
                    }
                )
                doc_chunks += 1
        print(f"  pages={len(pages)} text_pages={text_pages} chunks={doc_chunks}", flush=True)
    return chunks


def build_index(args: argparse.Namespace) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("Missing FAISS. Install it with: python3 -m pip install faiss-cpu") from exc

    chunks = build_chunks(args)
    if not chunks:
        raise SystemExit("No chunks were created. Try --extractor ocr and check Tesseract language data.")

    vectors = embed_texts([chunk["text"] for chunk in chunks], args.embedding_model, args.batch_size)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "embedding_model": args.embedding_model,
                "vector_dimensions": int(vectors.shape[1]),
                "pdf_dir": args.pdf_dir,
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote FAISS index to {index_dir / 'index.faiss'}")
    print(f"Wrote metadata to {index_dir / 'metadata.json'}")


def load_faiss_index(index_dir: Path):
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("Missing FAISS. Install it with: python3 -m pip install faiss-cpu") from exc

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise SystemExit(f"FAISS index not found in {index_dir}. Build it first with: python3 faiss_rag.py build")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, metadata


def search_index(args: argparse.Namespace) -> tuple[list[tuple[float, dict]], dict]:
    index, metadata = load_faiss_index(Path(args.index_dir))
    query_vector = embed_texts([args.question], metadata["embedding_model"], batch_size=1)
    scores, ids = index.search(query_vector, args.top_k)

    chunks = metadata["chunks"]
    results: list[tuple[float, dict]] = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id < 0:
            continue
        results.append((float(score), chunks[int(chunk_id)]))
    return results, metadata


def format_citation(chunk: dict) -> str:
    return f"{chunk['document']} p.{chunk['page']} chunk {chunk['chunk_on_page']}"


def compressed_context(results: list[tuple[float, dict]], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for rank, (_, chunk) in enumerate(results, start=1):
        text = chunk["text"]
        block = f"[{rank}] {format_citation(chunk)}\n{text}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= 0:
                break
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


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


def generate_answer(question: str, context: str, model: str, max_output_tokens: int) -> str:
    prompt = f"""You are a Vietnamese legal/regulatory RAG assistant.

Answer using only the retrieved PDF context below.
Write in Vietnamese.
Cite sources inline using bracket numbers like [1] or [2].
If the context does not contain enough information, say: "Không tìm thấy đủ thông tin trong tài liệu được cung cấp."

Question:
{question}

Retrieved context:
{context}
"""
    payload = openai_request(
        "responses",
        {"model": model, "input": prompt, "max_output_tokens": max_output_tokens},
    )
    text = extract_response_text(payload)
    if not text:
        raise RuntimeError("OpenAI API returned no text output.")
    return text


def ask_index(args: argparse.Namespace) -> None:
    if args.llm_model == "openai":
        raise SystemExit(
            "Invalid --llm-model value: 'openai'. Use --llm openai, and use a real model name for --llm-model."
        )

    results, _ = search_index(args)
    if not results:
        print("No matching chunks found.")
        return

    context = compressed_context(results, args.context_chars)
    if args.no_llm or args.llm == "none":
        print("\nRetrieved context")
        print("-----------------")
        print(context)
    else:
        print("\nLLM answer")
        print("----------")
        print(generate_answer(args.question, context, args.llm_model, args.max_output_tokens))

    print("\nSources")
    print("-------")
    for rank, (score, chunk) in enumerate(results, start=1):
        preview = clean_text(chunk["text"])[: args.preview_chars]
        print(f"[{rank}] score={score:.3f} {format_citation(chunk)}")
        print(f"    {preview}")


def inspect_index(args: argparse.Namespace) -> None:
    index, metadata = load_faiss_index(Path(args.index_dir))
    print(f"Index dir: {args.index_dir}")
    print(f"Chunks: {len(metadata['chunks'])}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Vector dimensions: {metadata['vector_dimensions']}")
    print(f"FAISS vectors: {index.ntotal}")


def add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Folder containing PDF files.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Where to store FAISS files.")
    parser.add_argument("--chunk-size", type=int, default=450, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap in words.")
    parser.add_argument("--min-page-chars", type=int, default=40, help="Skip pages with fewer characters.")
    parser.add_argument("--extractor", choices=["auto", "pdftotext", "ocr"], default="auto")
    parser.add_argument("--extract-timeout", type=int, default=10)
    parser.add_argument("--ocr-lang", default="vie+eng")
    parser.add_argument("--ocr-dpi", type=int, default=180)
    parser.add_argument("--ocr-timeout", type=int, default=240)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAISS + OpenAI PDF RAG.", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build FAISS index from PDFs.")
    add_build_args(build_parser)
    build_parser.set_defaults(func=build_index)

    ask_parser = subparsers.add_parser("ask", help="Ask using the FAISS index.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--context-chars", type=int, default=6000)
    ask_parser.add_argument("--preview-chars", type=int, default=420)
    ask_parser.add_argument("--llm", choices=["openai", "none"], default="openai", help="Generate an LLM answer.")
    ask_parser.add_argument("--no-llm", action="store_true", help="Only print retrieved chunks.")
    ask_parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL))
    ask_parser.add_argument("--max-output-tokens", type=int, default=900)
    ask_parser.set_defaults(func=ask_index)

    inspect_parser = subparsers.add_parser("inspect", help="Show FAISS index stats.")
    inspect_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    inspect_parser.set_defaults(func=inspect_index)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
