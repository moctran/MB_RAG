#!/usr/bin/env python3
"""Standalone Advanced PDF RAG.

This version does not import naive_rag.py and does not read rag_index.json.
It reads PDFs directly from pdf_files/ and stores its own index under
advanced_index/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


DEFAULT_PDF_DIR = Path("pdf_files")
DEFAULT_INDEX_PATH = Path("advanced_index/advanced_rag_index.json")
DEFAULT_OPENAI_MODEL = "gpt-5.5"
WORD_RE = re.compile(r"(?u)\b\w+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？;:])\s+|\n+")

DOMAIN_EXPANSIONS = {
    "dữ liệu cá nhân": ["bảo vệ dữ liệu cá nhân", "chủ thể dữ liệu", "xử lý dữ liệu cá nhân"],
    "du lieu ca nhan": ["bao ve du lieu ca nhan", "chu the du lieu", "xu ly du lieu ca nhan"],
    "kiểm soát nội bộ": ["hệ thống kiểm soát nội bộ", "kiểm toán nội bộ", "quản lý rủi ro"],
    "kiem soat noi bo": ["he thong kiem soat noi bo", "kiem toan noi bo", "quan ly rui ro"],
    "ngân hàng": ["tổ chức tín dụng", "ngân hàng nhà nước", "NHNN"],
    "ngan hang": ["to chuc tin dung", "ngan hang nha nuoc", "NHNN"],
    "thanh toán": ["dịch vụ thanh toán", "tài khoản thanh toán", "trung gian thanh toán"],
    "thanh toan": ["dich vu thanh toan", "tai khoan thanh toan", "trung gian thanh toan"],
    "dự phòng rủi ro": ["phân loại tài sản có", "trích lập dự phòng", "xử lý rủi ro"],
    "du phong rui ro": ["phan loai tai san co", "trich lap du phong", "xu ly rui ro"],
}

FOCUS_PHRASES = [
    "pham vi dieu chinh",
    "doi tuong ap dung",
    "giai thich tu ngu",
    "nguyen tac",
    "hieu luc thi hanh",
    "to chuc thuc hien",
    "quy dinh chuyen tiep",
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_vietnamese_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalized_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in WORD_RE.findall(text.lower()):
        if len(token) <= 1:
            continue
        tokens.append(token)
        unaccented = strip_vietnamese_accents(token)
        if unaccented != token:
            tokens.append(unaccented)
    return tokens


def canonical_tokens(text: str) -> list[str]:
    return [
        strip_vietnamese_accents(token)
        for token in WORD_RE.findall(text.lower())
        if len(token) > 1
    ]


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

    with tempfile.TemporaryDirectory(prefix="advanced_rag_ocr_") as temp_dir:
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


def extract_with_pdfplumber(pdf_path: Path) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((index, clean_text(text)))
    return pages


def extract_with_pypdf(pdf_path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path), strict=False)
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
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


def extract_pdf_pages(args: argparse.Namespace, pdf_path: Path) -> list[tuple[int, str]]:
    extractors = [args.extractor]
    if args.extractor == "auto":
        extractors = ["pdftotext", "pypdfium2", "pdfplumber", "pypdf", "ocr"]

    pages: list[tuple[int, str]] = []
    errors: list[str] = []
    for extractor in extractors:
        try:
            if extractor == "pdftotext":
                pages = extract_with_pdftotext(pdf_path, args.extract_timeout)
            elif extractor == "ocr":
                pages = extract_with_ocr(pdf_path, args.ocr_timeout, args.ocr_dpi, args.ocr_lang)
            elif extractor in {"pypdfium2", "pdfplumber", "pypdf"}:
                pages = extract_with_python_timeout(extractor, pdf_path, args.extract_timeout)
            else:
                raise ValueError(f"Unknown extractor: {extractor}")
        except Exception as exc:
            errors.append(f"{extractor}: {exc}")
            continue

        if sum(len(text) for _, text in pages) > 0:
            return pages
        errors.append(f"{extractor}: extracted no text")

    if errors:
        print(f"  warning: extraction issues for {pdf_path.name}: {'; '.join(errors)}")
    return pages


def char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    compact = re.sub(r"\s+", " ", strip_vietnamese_accents(text.lower())).strip()
    grams: list[str] = []
    for n in range(min_n, max_n + 1):
        grams.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return [gram for gram in grams if len(gram.strip()) == len(gram)]


def semantic_features(text: str) -> Counter[str]:
    tokens = normalized_tokens(text)
    features: Counter[str] = Counter()
    for token in tokens:
        features[f"w:{token}"] += 1

    for left, right in zip(tokens, tokens[1:]):
        features[f"b:{left}_{right}"] += 1

    for gram in char_ngrams(text):
        features[f"c:{gram}"] += 0.2

    return features


def stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def sparse_tfidf_vector(features: Counter[str], feature_df: dict[str, int], total_docs: int, dims: int) -> list[list[float]]:
    values: dict[int, float] = {}
    for feature, tf in features.items():
        df = feature_df.get(feature, 0)
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        hashed = stable_hash(feature)
        index = hashed % dims
        sign = 1 if (hashed >> 63) == 0 else -1
        values[index] = values.get(index, 0.0) + sign * (1 + math.log(float(tf))) * idf

    norm = math.sqrt(sum(value * value for value in values.values()))
    if norm == 0:
        return []
    return [[index, value / norm] for index, value in sorted(values.items()) if value]


def dot_sparse(left: list[list[float]], right: list[list[float]]) -> float:
    right_values = {int(index): float(value) for index, value in right}
    return sum(float(value) * right_values.get(int(index), 0.0) for index, value in left)


def rewrite_query(query: str) -> list[tuple[str, float, str]]:
    variants: list[tuple[str, float, str]] = [(clean_text(query), 1.0, "original")]
    unaccented = strip_vietnamese_accents(query)
    if unaccented != query:
        variants.append((unaccented, 0.95, "unaccented"))

    lowered = query.lower()
    lowered_unaccented = unaccented.lower()
    expansions: list[str] = []
    for trigger, related_terms in DOMAIN_EXPANSIONS.items():
        if trigger in lowered or trigger in lowered_unaccented:
            expansions.extend(related_terms)

    if expansions:
        variants.append((f"{query} {' '.join(expansions)}", 0.55, "expanded"))
        variants.append(
            (
                f"{unaccented} {' '.join(strip_vietnamese_accents(term) for term in expansions)}",
                0.50,
                "expanded_unaccented",
            )
        )

    deduped: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    for variant, weight, label in variants:
        normalized = variant.lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append((variant, weight, label))
    return deduped


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}. Build it first with: python3 advanced_rag.py build")
    return json.loads(index_path.read_text(encoding="utf-8"))


def format_citation(chunk: dict) -> str:
    return f"{chunk['document']} p.{chunk['page']} chunk {chunk['chunk_on_page']}"


def best_sentences(query: str, results: list[tuple[float, dict]], max_sentences: int) -> list[str]:
    query_terms = set(normalized_tokens(query))
    selected: list[str] = []
    seen: set[str] = set()

    for _, chunk in results:
        sentences = [clean_text(part) for part in SENTENCE_RE.split(chunk["text"]) if clean_text(part)]
        if not sentences:
            sentences = [chunk["text"]]

        ranked = []
        for sentence in sentences:
            sentence_terms = set(normalized_tokens(sentence))
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


def call_openai_llm(question: str, context: str, model: str, max_output_tokens: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

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


def build_advanced_index(args: argparse.Namespace) -> None:
    pdfs = discover_pdfs(Path(args.pdf_dir))
    raw_chunks: list[dict] = []
    document_stats: list[dict[str, int | str]] = []
    skipped_pages = 0

    for pdf_path in pdfs:
        print(f"Reading {pdf_path.name} ...", flush=True)
        pages = extract_pdf_pages(args, pdf_path)
        doc_text_pages = 0
        doc_chunks = 0

        for page_number, page_text in pages:
            if len(page_text) < args.min_page_chars:
                skipped_pages += 1
                continue

            doc_text_pages += 1
            for chunk_on_page, text in enumerate(chunk_words(page_text, args.chunk_size, args.overlap), start=1):
                if not normalized_tokens(text):
                    continue
                raw_chunks.append(
                    {
                        "id": len(raw_chunks),
                        "source": str(pdf_path),
                        "document": pdf_path.name,
                        "page": page_number,
                        "chunk_on_page": chunk_on_page,
                        "text": text,
                    }
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

    if not raw_chunks:
        raise SystemExit("No text chunks were created. Try --extractor ocr and install the right Tesseract language data.")

    chunk_features = [semantic_features(chunk["text"]) for chunk in raw_chunks]
    feature_df: Counter[str] = Counter()
    for features in chunk_features:
        feature_df.update(features.keys())

    chunks = []
    for chunk, features in zip(raw_chunks, chunk_features):
        terms = Counter(normalized_tokens(chunk["text"]))
        chunks.append(
            {
                **chunk,
                "term_counts": dict(terms),
                "length": sum(terms.values()),
                "vector": sparse_tfidf_vector(features, feature_df, len(raw_chunks), args.vector_dims),
            }
        )

    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(chunk["term_counts"].keys())

    avgdl = sum(chunk["length"] for chunk in chunks) / len(chunks)
    index = {
        "version": 2,
        "source": "standalone_advanced_rag",
        "settings": {
            "pdf_dir": args.pdf_dir,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "min_page_chars": args.min_page_chars,
            "extractor": args.extractor,
            "ocr_lang": args.ocr_lang,
            "ocr_dpi": args.ocr_dpi,
            "vector_dims": args.vector_dims,
            "note": "Hashed TF-IDF vectors, not a trained embedding model.",
        },
        "stats": {
            "documents": len(pdfs),
            "chunks": len(chunks),
            "avg_chunk_terms": avgdl,
            "skipped_pages": skipped_pages,
        },
        "document_stats": document_stats,
        "doc_freq": dict(doc_freq),
        "feature_df": dict(feature_df),
        "chunks": chunks,
    }

    output_path = Path(args.index_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote standalone advanced index with {len(chunks)} chunks to {output_path}")


def bm25_score_chunks(index: dict, query: str) -> dict[int, float]:
    query_terms = Counter(normalized_tokens(query))
    if not query_terms:
        return {}

    chunks = index["chunks"]
    doc_freq = index["doc_freq"]
    total_chunks = len(chunks)
    avgdl = float(index["stats"]["avg_chunk_terms"])
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}

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
            scores[int(chunk["id"])] = score
    return scores


def semantic_score_chunks(index: dict, query: str) -> dict[int, float]:
    feature_df = index["feature_df"]
    dims = int(index["settings"]["vector_dims"])
    query_vector = sparse_tfidf_vector(semantic_features(query), feature_df, len(index["chunks"]), dims)
    if not query_vector:
        return {}
    return {int(chunk["id"]): dot_sparse(query_vector, chunk["vector"]) for chunk in index["chunks"]}


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def phrase_bonus(query: str, text: str) -> float:
    query_norm = strip_vietnamese_accents(query.lower())
    text_norm = strip_vietnamese_accents(text.lower())
    query_terms = canonical_tokens(query)
    bonus = 0.0

    if query_norm and query_norm in text_norm:
        bonus += 0.60

    for n, weight in ((4, 0.08), (3, 0.05), (2, 0.02)):
        for start in range(0, max(0, len(query_terms) - n + 1)):
            phrase = " ".join(query_terms[start : start + n])
            if phrase in text_norm:
                bonus += weight

    return min(bonus, 0.45)


def focus_bonus(query: str, text: str) -> float:
    query_norm = strip_vietnamese_accents(query.lower())
    text_norm = strip_vietnamese_accents(text.lower())
    bonus = 0.0
    for phrase in FOCUS_PHRASES:
        if phrase in query_norm and phrase in text_norm:
            bonus += 2.0
    return bonus


def hybrid_retrieve(args: argparse.Namespace, index: dict) -> tuple[list[tuple[str, float, str]], list[tuple[float, dict, dict[str, float]]]]:
    queries = rewrite_query(args.question)
    combined_bm25: dict[int, float] = {}
    combined_semantic: dict[int, float] = {}

    for query, query_weight, _ in queries:
        bm25_scores = bm25_score_chunks(index, query)
        semantic_scores = semantic_score_chunks(index, query)
        for chunk_id, score in bm25_scores.items():
            combined_bm25[chunk_id] = max(combined_bm25.get(chunk_id, 0.0), score * query_weight)
        for chunk_id, score in semantic_scores.items():
            combined_semantic[chunk_id] = max(combined_semantic.get(chunk_id, 0.0), score * query_weight)

    normalized_bm25 = minmax(combined_bm25)
    normalized_semantic = minmax(combined_semantic)
    chunks_by_id = {int(chunk["id"]): chunk for chunk in index["chunks"]}
    candidate_ids = set(normalized_bm25) | set(normalized_semantic)
    scored = []

    for chunk_id in candidate_ids:
        chunk = chunks_by_id[chunk_id]
        bm25 = normalized_bm25.get(chunk_id, 0.0)
        semantic = normalized_semantic.get(chunk_id, 0.0)
        bonus = phrase_bonus(args.question, chunk["text"])
        focused = focus_bonus(args.question, chunk["text"])
        score = args.keyword_weight * bm25 + args.semantic_weight * semantic + bonus + focused
        scored.append((score, chunk, {"bm25": bm25, "semantic": semantic, "phrase_bonus": bonus, "focus_bonus": focused}))

    scored.sort(key=lambda item: item[0], reverse=True)
    return queries, scored[: args.candidates]


def rerank(args: argparse.Namespace, retrieved: list[tuple[float, dict, dict[str, float]]]) -> list[tuple[float, dict, dict[str, float]]]:
    query_terms = set(normalized_tokens(args.question))
    reranked = []
    for initial_score, chunk, components in retrieved:
        text_terms = set(normalized_tokens(chunk["text"]))
        coverage = len(query_terms & text_terms) / max(1, len(query_terms))
        density = len(query_terms & text_terms) / max(1, len(text_terms))
        exact_bonus = phrase_bonus(args.question, chunk["text"])
        focused = focus_bonus(args.question, chunk["text"])
        final_score = initial_score + args.coverage_weight * coverage + args.density_weight * density + exact_bonus + focused
        enriched = dict(components)
        enriched.update({"coverage": coverage, "density": density, "exact_bonus": exact_bonus, "focus_rerank": focused})
        reranked.append((final_score, chunk, enriched))
    reranked.sort(key=lambda item: item[0], reverse=True)
    return reranked[: args.top_k]


def compressed_context(question: str, results: list[tuple[float, dict, dict[str, float]]], max_chars: int) -> str:
    used = 0
    selected: list[str] = []

    for rank, (score, chunk, _) in enumerate(results, start=1):
        sentences = best_sentences(question, [(score, chunk)], max_sentences=8)
        for sentence in sentences:
            line = f"[{rank}] {sentence}"
            addition = len(line) + 3
            if used + addition > max_chars:
                return "\n".join(f"- {item}" for item in selected)
            selected.append(line)
            used += addition

    return "\n".join(f"- {item}" for item in selected)


def llm_context(context: str, results: list[tuple[float, dict, dict[str, float]]], max_chars: int) -> str:
    blocks = []
    used = 0
    for rank, (_, chunk, _) in enumerate(results, start=1):
        block = f"[{rank}] {format_citation(chunk)}\n{chunk['text']}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= 0:
                break
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
    compressed = f"Compressed relevant sentences:\n{context}"
    if used + len(compressed) <= max_chars:
        blocks.insert(0, compressed)
    return "\n\n".join(blocks)


def ask_advanced(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index_path))
    rewritten_queries, candidates = hybrid_retrieve(args, index)
    if not candidates:
        print("No matching chunks found.")
        return

    results = rerank(args, candidates)
    context = compressed_context(args.question, results, args.max_context_chars)

    print("\nQuery rewrites")
    print("--------------")
    for query, weight, label in rewritten_queries:
        print(f"- ({label}, weight={weight:.2f}) {query}")

    if args.llm == "openai":
        print("\nLLM answer")
        print("----------")
        try:
            print(
                call_openai_llm(
                    args.question,
                    llm_context(context, results, args.llm_context_chars),
                    args.llm_model,
                    args.llm_max_output_tokens,
                )
            )
        except RuntimeError as exc:
            print(f"LLM error: {exc}")
            print("\nCompressed extractive answer")
            print("---------------------------")
            print(context)
    else:
        print("\nCompressed extractive answer")
        print("---------------------------")
        print(context)

    print("\nSources")
    print("-------")
    for rank, (score, chunk, components) in enumerate(results, start=1):
        preview = clean_text(chunk["text"])[: args.preview_chars]
        detail = " ".join(f"{key}={value:.3f}" for key, value in components.items())
        print(f"[{rank}] score={score:.3f} {detail} {format_citation(chunk)}")
        print(f"    {preview}")


def inspect_advanced(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index_path))
    stats = index["stats"]
    print(f"Source: {index.get('source', 'unknown')}")
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Vector dimensions: {index['settings']['vector_dims']}")
    print(f"Average chunk terms: {stats['avg_chunk_terms']:.1f}")
    print(f"Index path: {args.index_path}")


def add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Folder containing PDF files.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Output advanced index path.")
    parser.add_argument("--chunk-size", type=int, default=450, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap in words.")
    parser.add_argument("--min-page-chars", type=int, default=40, help="Skip pages with little text.")
    parser.add_argument(
        "--extractor",
        choices=["auto", "pdftotext", "pypdfium2", "pypdf", "pdfplumber", "ocr"],
        default="auto",
        help="PDF text extractor to use.",
    )
    parser.add_argument("--extract-timeout", type=int, default=10, help="Timeout in seconds per non-OCR extractor.")
    parser.add_argument("--ocr-lang", default="eng", help="Tesseract language code, for example eng or vie+eng.")
    parser.add_argument("--ocr-dpi", type=int, default=120, help="DPI used when rendering PDFs for OCR.")
    parser.add_argument("--ocr-timeout", type=int, default=180, help="Timeout in seconds for OCR rendering/OCR.")
    parser.add_argument("--vector-dims", type=int, default=2048, help="Hashed vector dimensions.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone advanced PDF RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build an independent advanced index from PDFs.")
    add_build_args(build_parser)
    build_parser.set_defaults(func=build_advanced_index)

    ask_parser = subparsers.add_parser("ask", help="Ask using advanced hybrid retrieval.")
    ask_parser.add_argument("question", help="Question to search for.")
    ask_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to advanced index.")
    ask_parser.add_argument("--candidates", type=int, default=30, help="Candidate chunks before re-ranking.")
    ask_parser.add_argument("--top-k", type=int, default=6, help="Final chunks after re-ranking.")
    ask_parser.add_argument("--keyword-weight", type=float, default=0.45, help="BM25 contribution.")
    ask_parser.add_argument("--semantic-weight", type=float, default=0.55, help="Semantic vector contribution.")
    ask_parser.add_argument("--coverage-weight", type=float, default=0.25, help="Re-rank query coverage contribution.")
    ask_parser.add_argument("--density-weight", type=float, default=0.10, help="Re-rank term density contribution.")
    ask_parser.add_argument("--max-context-chars", type=int, default=4500, help="Compressed context character budget.")
    ask_parser.add_argument("--preview-chars", type=int, default=420, help="Characters shown per source.")
    ask_parser.add_argument("--llm", choices=["none", "openai"], default="none", help="Use an LLM to write the answer.")
    ask_parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    ask_parser.add_argument("--llm-max-output-tokens", type=int, default=900)
    ask_parser.add_argument("--llm-context-chars", type=int, default=2500)
    ask_parser.set_defaults(func=ask_advanced)

    inspect_parser = subparsers.add_parser("inspect", help="Show advanced index statistics.")
    inspect_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    inspect_parser.set_defaults(func=inspect_advanced)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
