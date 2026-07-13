#!/usr/bin/env python3
"""Standalone local GraphRAG over PDFs in pdf_files/.

This version builds a lightweight knowledge graph without external graph
databases:
- extract and chunk PDF text
- extract legal/domain entities with deterministic heuristics
- connect entities by chunk co-occurrence
- group entities into graph communities with weighted label propagation
- answer with local entity-neighborhood search or global community search
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
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PDF_DIR = Path("pdf_files")
DEFAULT_INDEX_PATH = Path("graph_index/graph_rag_index.json")
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OCR_LANG = "vie+eng"
DEFAULT_OCR_DPI = 180
DEFAULT_OCR_PSM = "4"
WORD_RE = re.compile(r"(?u)\b\w+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？;:])\s+|\n+")

LEGAL_REFERENCE_RE = re.compile(
    r"(?iu)\b(?:"
    r"Điều\s+\d+[a-z]?|Khoản\s+\d+[a-z]?|Mục\s+\d+[a-z]?|Chương\s+[IVXLC\d]+|"
    r"Nghị\s+định\s+\d+/\d{4}/[A-ZĐ/-]+|"
    r"Thông\s+tư\s+\d+/\d{4}/[A-ZĐ/-]+|"
    r"Quyết\s+định\s+\d+/\d{4}/[A-ZĐ/-]+|"
    r"Luật\s+[A-ZÀ-Ỹa-zà-ỹĐđ\s]{3,60}"
    r")\b"
)
ABBREVIATION_RE = re.compile(r"(?u)\b[A-ZĐ]{2,}(?:[-/][A-Z0-9Đ]+)*\b")
CAPITALIZED_PHRASE_RE = re.compile(
    r"(?u)\b[A-ZÀ-ỸĐ][\wÀ-ỹĐđ/-]*"
    r"(?:\s+(?:[A-ZÀ-ỸĐ0-9][\wÀ-ỹĐđ/-]*|và|của|cho|về|trong|theo|tại)){0,5}"
)

DOMAIN_PHRASES = [
    "bảo vệ dữ liệu cá nhân",
    "chủ thể dữ liệu",
    "dữ liệu cá nhân",
    "dịch vụ thanh toán",
    "đối tượng áp dụng",
    "dự phòng rủi ro",
    "hệ thống kiểm soát nội bộ",
    "kiểm soát nội bộ",
    "kiểm toán nội bộ",
    "ngân hàng nhà nước",
    "nguyên tắc",
    "phạm vi điều chỉnh",
    "phân loại tài sản có",
    "quản lý rủi ro",
    "tài khoản thanh toán",
    "thanh toán không dùng tiền mặt",
    "tổ chức cung ứng dịch vụ thanh toán",
    "tổ chức tín dụng",
    "trích lập dự phòng",
    "trung gian thanh toán",
    "xử lý dữ liệu cá nhân",
    "xử lý rủi ro",
]

GLOBAL_QUERY_HINTS = {
    "bao quát",
    "chung",
    "comprehensive",
    "global",
    "khái quát",
    "overview",
    "so sánh",
    "summary",
    "toàn bộ",
    "toàn diện",
    "tóm tắt",
    "tổng hợp",
    "tổng quan",
}

ENTITY_STOP_KEYS = {
    "ban",
    "bao",
    "ben kiem",
    "bo",
    "ca nhan",
    "can cu",
    "chu",
    "chinh",
    "chuong",
    "co",
    "cong hoa",
    "cua",
    "dang",
    "danh",
    "dich",
    "dieu",
    "dinh",
    "dinh cua",
    "dinh tai",
    "dinh ve",
    "doc lap",
    "dong",
    "duoc",
    "giay",
    "hanh phuc",
    "kinh",
    "lieu",
    "muc",
    "ngan",
    "ngay",
    "nhan",
    "quoc",
    "so",
    "theo",
    "the",
    "thong",
    "tu do",
}

GLOBAL_FILLER_TERMS = {
    "cac",
    "chinh",
    "cua",
    "diem",
    "noi",
    "quy",
    "quy dinh",
    "tai lieu",
    "tom tat",
    "tong hop",
    "tong quan",
    "van ban",
}

OCR_TEXT_REPLACEMENTS = [
    (r"\bdu\s+li[eệ]u\b", "dữ liệu"),
    (r"\bdu\s+ligu\b", "dữ liệu"),
    (r"\bdir\s+ligu\b", "dữ liệu"),
    (r"\bdit\s+li[eệ]u\b", "dữ liệu"),
    (r"\bdit\s+ligu\b", "dữ liệu"),
    (r"\bd[ữừư]\s+li[eệ]u\b", "dữ liệu"),
    (r"\bd[ée]\s+li[eệ]u\b", "dữ liệu"),
    (r"\bd[eé]\s+ligu\b", "dữ liệu"),
    (r"\bch[uủ]\s+th[eé]\b", "chủ thể"),
    (r"\bchi\s+th[eé]\b", "chủ thể"),
    (r"\bcht\s+th[eé]\b", "chủ thể"),
    (r"\bx[uử]\s+l[yý]\b", "xử lý"),
    (r"\bxi\s+ly\b", "xử lý"),
    (r"\bxtr\s+ly\b", "xử lý"),
    (r"\bki[eé]m\s+so[aá]t\b", "kiểm soát"),
    (r"\bki[eé]m\s+so[aá]t\s+v[aà]\s+x[uử]\s+l[yý]\b", "kiểm soát và xử lý"),
    (r"\bh[oỗ]\s+s[oơ]\b", "hồ sơ"),
    (r"\bthay\s+doi\b", "thay đổi"),
    (r"\bs[uử]\s+dung\b", "sử dụng"),
    (r"\bduge\b", "được"),
    (r"\bdur?gc\b", "được"),
    (r"\bquy[eé]n\b", "quyền"),
    (r"\bd[eé]ng\s+y\b", "đồng ý"),
    (r"\bmat\s+d[ữư]\s+li[eệ]u\b", "mất dữ liệu"),
    (r"\bl[oộ],\s*mat\b", "lộ, mất"),
    (r"\bkh[aả]\s+nang\b", "khả năng"),
    (r"\bt[oô]\s+ch[uứ]c\b", "tổ chức"),
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_vietnamese_ocr_text(text: str) -> str:
    text = clean_text(text)
    text = text.replace("Ð", "Đ").replace("ð", "đ")
    text = re.sub(r"\b([A-ZÀ-ỸĐ])\s+([A-ZÀ-ỸĐ])\s+([A-ZÀ-ỸĐ])\b", r"\1\2\3", text)
    for pattern, replacement in OCR_TEXT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return clean_text(text)


def strip_vietnamese_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalized_tokens(text: str) -> list[str]:
    return [
        strip_vietnamese_accents(token.lower())
        for token in WORD_RE.findall(text)
        if len(token) > 1
    ]


def normalize_entity(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"[,:;.!?()\[\]{}]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return strip_vietnamese_accents(text.lower())


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

    candidates = [
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin" / name,
        Path("/opt/homebrew/bin") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
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


def extract_with_ocr(pdf_path: Path, timeout: int, dpi: int, lang: str, psm: str) -> list[tuple[int, str]]:
    pdftoppm = find_executable("pdftoppm")
    tesseract = find_executable("tesseract")
    if not pdftoppm:
        raise FileNotFoundError("pdftoppm was not found")
    if not tesseract:
        raise FileNotFoundError("tesseract was not found")

    with tempfile.TemporaryDirectory(prefix="graph_rag_ocr_") as temp_dir:
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
                [
                    tesseract,
                    str(image_path),
                    "stdout",
                    "-l",
                    lang,
                    "--psm",
                    psm,
                    "-c",
                    "preserve_interword_spaces=1",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            pages.append((page_number_from_image(image_path), repair_vietnamese_ocr_text(completed.stdout)))
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
                pages = extract_with_ocr(
                    pdf_path,
                    args.ocr_timeout,
                    args.ocr_dpi,
                    args.ocr_lang,
                    args.ocr_psm,
                )
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
        print(f"  warning: extraction issues for {pdf_path.name}: {'; '.join(errors)}", file=sys.stderr)
    return pages


def extract_entities(text: str, max_entities: int) -> list[str]:
    candidates: list[str] = []

    candidates.extend(match.group(0) for match in LEGAL_REFERENCE_RE.finditer(text))
    candidates.extend(match.group(0) for match in ABBREVIATION_RE.finditer(text))

    normalized_text = strip_vietnamese_accents(text.lower())
    for phrase in DOMAIN_PHRASES:
        if strip_vietnamese_accents(phrase) in normalized_text:
            candidates.append(phrase)

    for match in CAPITALIZED_PHRASE_RE.finditer(text):
        candidate = clean_text(match.group(0))
        if 4 <= len(candidate) <= 90:
            candidates.append(candidate)

    counts: Counter[str] = Counter()
    display_by_key: dict[str, str] = {}
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip(" -:;,.")
        key = normalize_entity(candidate)
        tokens = normalized_tokens(candidate)
        if not key or key in ENTITY_STOP_KEYS or len(key) < 3 or len(tokens) > 12:
            continue
        is_abbreviation = bool(ABBREVIATION_RE.fullmatch(candidate))
        is_legal_reference = bool(LEGAL_REFERENCE_RE.fullmatch(candidate))
        is_domain_phrase = any(key == normalize_entity(phrase) for phrase in DOMAIN_PHRASES)
        is_uppercase_noise = (
            candidate.upper() == candidate
            and " " in candidate
            and not any("À" <= char <= "ỹ" or char == "Đ" for char in candidate)
        )
        if is_uppercase_noise and not (is_legal_reference or is_domain_phrase):
            continue
        if len(tokens) == 1 and not (is_abbreviation or is_legal_reference or is_domain_phrase):
            continue
        counts[key] += 1
        if key not in display_by_key or len(candidate) < len(display_by_key[key]):
            display_by_key[key] = candidate

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [display_by_key[key] for key, _ in ranked[:max_entities]]


def build_graph_index(args: argparse.Namespace) -> None:
    pdfs = discover_pdfs(Path(args.pdf_dir))
    chunks: list[dict] = []
    document_stats: list[dict[str, int | str]] = []
    entity_mentions: dict[str, Counter[str]] = defaultdict(Counter)
    entity_chunks: dict[str, Counter[int]] = defaultdict(Counter)
    edge_weights: Counter[tuple[str, str]] = Counter()
    edge_chunks: dict[tuple[str, str], set[int]] = defaultdict(set)
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
                terms = Counter(normalized_tokens(text))
                if not terms:
                    continue

                entities = extract_entities(text, args.max_entities_per_chunk)
                entity_keys = []
                for entity in entities:
                    key = normalize_entity(entity)
                    if key and key not in entity_keys:
                        entity_keys.append(key)
                        entity_mentions[key][entity] += 1
                        entity_chunks[key][len(chunks)] += 1

                limited_keys = entity_keys[: args.max_graph_entities_per_chunk]
                if len(limited_keys) > 1:
                    increment = 1.0 / max(1, len(limited_keys) - 1)
                    for left_index, left in enumerate(limited_keys):
                        for right in limited_keys[left_index + 1 :]:
                            edge = tuple(sorted((left, right)))
                            edge_weights[edge] += increment
                            edge_chunks[edge].add(len(chunks))

                chunks.append(
                    {
                        "id": len(chunks),
                        "source": str(pdf_path),
                        "document": pdf_path.name,
                        "page": page_number,
                        "chunk_on_page": chunk_on_page,
                        "text": text,
                        "term_counts": dict(terms),
                        "length": sum(terms.values()),
                        "entities": entity_keys,
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

    if not chunks:
        raise SystemExit("No text chunks were created. Try --extractor ocr and install the right Tesseract language data.")

    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(chunk["term_counts"].keys())

    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for (left, right), weight in edge_weights.items():
        adjacency[left][right] = float(weight)
        adjacency[right][left] = float(weight)

    entities = {}
    for key, mentions in entity_mentions.items():
        name = mentions.most_common(1)[0][0]
        degree = sum(adjacency.get(key, {}).values())
        entities[key] = {
            "name": name,
            "mentions": int(sum(mentions.values())),
            "chunks": {str(chunk_id): count for chunk_id, count in entity_chunks[key].items()},
            "degree": degree,
        }

    communities = detect_communities(adjacency, entities, chunks, args.community_iterations, args.community_top_chunks)

    avgdl = sum(chunk["length"] for chunk in chunks) / len(chunks)
    relationships = [
        {
            "source": left,
            "target": right,
            "weight": float(weight),
            "chunks": sorted(edge_chunks[(left, right)])[: args.max_edge_chunk_refs],
        }
        for (left, right), weight in edge_weights.most_common()
    ]

    index = {
        "version": 1,
        "source": "standalone_graph_rag",
        "settings": {
            "pdf_dir": args.pdf_dir,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "min_page_chars": args.min_page_chars,
            "extractor": args.extractor,
            "ocr_lang": args.ocr_lang,
            "ocr_dpi": args.ocr_dpi,
            "ocr_psm": args.ocr_psm,
            "max_entities_per_chunk": args.max_entities_per_chunk,
            "community_iterations": args.community_iterations,
        },
        "stats": {
            "documents": len(pdfs),
            "chunks": len(chunks),
            "entities": len(entities),
            "relationships": len(relationships),
            "communities": len(communities),
            "avg_chunk_terms": avgdl,
            "skipped_pages": skipped_pages,
        },
        "document_stats": document_stats,
        "doc_freq": dict(doc_freq),
        "entities": entities,
        "relationships": relationships,
        "communities": communities,
        "chunks": chunks,
    }

    output_path = Path(args.index_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote GraphRAG index to {output_path}")
    print(f"Chunks={len(chunks)} entities={len(entities)} relationships={len(relationships)} communities={len(communities)}")


def detect_communities(
    adjacency: dict[str, dict[str, float]],
    entities: dict[str, dict],
    chunks: list[dict],
    iterations: int,
    top_chunks: int,
) -> list[dict]:
    labels = {key: key for key in entities}
    for _ in range(iterations):
        changed = False
        for node in sorted(entities, key=lambda key: (-entities[key]["degree"], key)):
            label_scores: Counter[str] = Counter()
            for neighbor, weight in adjacency.get(node, {}).items():
                label_scores[labels[neighbor]] += weight
            if not label_scores:
                continue
            best_label, _ = max(label_scores.items(), key=lambda item: (item[1], -len(item[0])))
            if best_label != labels[node]:
                labels[node] = best_label
                changed = True
        if not changed:
            break

    grouped: dict[str, list[str]] = defaultdict(list)
    for node, label in labels.items():
        grouped[label].append(node)

    chunk_by_id = {int(chunk["id"]): chunk for chunk in chunks}
    communities = []
    for community_id, (_, members) in enumerate(
        sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
        start=1,
    ):
        member_set = set(members)
        chunk_scores: Counter[int] = Counter()
        for entity_key in members:
            for chunk_id, count in entities[entity_key]["chunks"].items():
                chunk_scores[int(chunk_id)] += count

        top_entity_keys = sorted(
            members,
            key=lambda key: (-entities[key]["mentions"], -entities[key]["degree"], entities[key]["name"]),
        )[:12]
        top_chunk_ids = [chunk_id for chunk_id, _ in chunk_scores.most_common(top_chunks)]
        excerpts = []
        for chunk_id in top_chunk_ids[:3]:
            chunk = chunk_by_id[chunk_id]
            repaired_text = repair_vietnamese_ocr_text(chunk["text"])
            sentences = [clean_text(part) for part in SENTENCE_RE.split(repaired_text) if clean_text(part)]
            excerpts.append(sentences[0] if sentences else clean_text(chunk["text"])[:240])

        communities.append(
            {
                "id": community_id,
                "entities": sorted(member_set),
                "top_entities": [entities[key]["name"] for key in top_entity_keys],
                "top_chunk_ids": top_chunk_ids,
                "summary": summarize_community([entities[key]["name"] for key in top_entity_keys], excerpts),
            }
        )
    return communities


def summarize_community(top_entities: list[str], excerpts: list[str]) -> str:
    entity_text = ", ".join(top_entities[:10]) if top_entities else "no named entities"
    excerpt_text = " ".join(excerpts)
    if len(excerpt_text) > 900:
        excerpt_text = excerpt_text[:900].rsplit(" ", 1)[0]
    return f"Entities: {entity_text}\nKey excerpts: {excerpt_text}"


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}. Build it first with: python3 graph_rag.py build")
    return json.loads(index_path.read_text(encoding="utf-8"))


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


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def text_overlap_score(query: str, text: str) -> float:
    query_terms = set(normalized_tokens(query))
    text_terms = set(normalized_tokens(text))
    if not query_terms or not text_terms:
        return 0.0
    overlap = len(query_terms & text_terms) / max(1, len(query_terms))
    query_norm = strip_vietnamese_accents(query.lower())
    text_norm = strip_vietnamese_accents(text.lower())
    phrase = 0.35 if query_norm and query_norm in text_norm else 0.0
    return overlap + phrase


def score_entities(index: dict, query: str, limit: int) -> list[tuple[float, str]]:
    scored = []
    for key, entity in index["entities"].items():
        name_score = text_overlap_score(query, entity["name"])
        if key in strip_vietnamese_accents(query.lower()):
            name_score += 0.6
        if name_score > 0:
            scored.append((name_score + min(float(entity["mentions"]) / 40.0, 0.25), key))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def adjacency_from_relationships(index: dict) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in index["relationships"]:
        left = edge["source"]
        right = edge["target"]
        weight = float(edge["weight"])
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    return adjacency


def choose_mode(mode: str, query: str) -> str:
    if mode != "auto":
        return mode
    query_norm = set(normalized_tokens(query))
    return "global" if query_norm & {strip_vietnamese_accents(term) for term in GLOBAL_QUERY_HINTS} else "local"


def global_scoring_query(query: str) -> str:
    normalized = strip_vietnamese_accents(query.lower())
    for phrase in sorted(GLOBAL_FILLER_TERMS | GLOBAL_QUERY_HINTS, key=len, reverse=True):
        normalized = normalized.replace(strip_vietnamese_accents(phrase), " ")
    return " ".join(normalized_tokens(normalized))


def local_retrieve(args: argparse.Namespace, index: dict) -> tuple[list[tuple[float, dict, dict[str, float]]], list[str]]:
    chunks_by_id = {int(chunk["id"]): chunk for chunk in index["chunks"]}
    adjacency = adjacency_from_relationships(index)
    seed_scores = score_entities(index, args.question, args.seed_entities)
    bm25_scores = bm25_score_chunks(index, args.question)

    if not seed_scores and bm25_scores:
        for chunk_id, _ in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[: args.top_k]:
            for entity_key in chunks_by_id[chunk_id].get("entities", [])[:3]:
                seed_scores.append((0.35, entity_key))

    graph_scores: Counter[int] = Counter()
    seed_names = []
    for seed_score, entity_key in seed_scores:
        entity = index["entities"][entity_key]
        seed_names.append(entity["name"])
        for chunk_id, count in entity["chunks"].items():
            graph_scores[int(chunk_id)] += seed_score * count
        for neighbor, weight in sorted(adjacency.get(entity_key, {}).items(), key=lambda item: item[1], reverse=True)[: args.neighbors]:
            neighbor_entity = index["entities"][neighbor]
            for chunk_id, count in neighbor_entity["chunks"].items():
                graph_scores[int(chunk_id)] += seed_score * weight * count * 0.35

    for chunk_id, score in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[: args.candidates]:
        graph_scores[chunk_id] += score * 0.2

    normalized_graph = minmax(dict(graph_scores))
    normalized_bm25 = minmax(bm25_scores)
    candidate_ids = set(normalized_graph) | set(sorted(bm25_scores, key=bm25_scores.get, reverse=True)[: args.candidates])
    results = []
    for chunk_id in candidate_ids:
        chunk = chunks_by_id[chunk_id]
        graph = normalized_graph.get(chunk_id, 0.0)
        keyword = normalized_bm25.get(chunk_id, 0.0)
        phrase = text_overlap_score(args.question, chunk["text"])
        score = args.graph_weight * graph + args.keyword_weight * keyword + 0.25 * phrase
        results.append((score, chunk, {"graph": graph, "keyword": keyword, "phrase": phrase}))
    results.sort(key=lambda item: item[0], reverse=True)
    return results[: args.top_k], seed_names[: args.seed_entities]


def global_retrieve(args: argparse.Namespace, index: dict) -> tuple[list[tuple[float, dict, dict[str, float]]], list[dict]]:
    chunks_by_id = {int(chunk["id"]): chunk for chunk in index["chunks"]}
    community_scores = []
    scoring_query = global_scoring_query(args.question)
    for community in index["communities"]:
        size_prior = min(len(community["entities"]) / 80.0, 0.35)
        if scoring_query:
            score = text_overlap_score(scoring_query, community["summary"]) + size_prior
        else:
            score = size_prior
        if score > 0:
            community_scores.append((score, community))

    community_scores.sort(key=lambda item: item[0], reverse=True)
    selected_communities = [community for _, community in community_scores[: args.communities]]

    bm25_scores = bm25_score_chunks(index, args.question)
    chunk_scores: Counter[int] = Counter()
    for community_score, community in community_scores[: args.communities]:
        for rank, chunk_id in enumerate(community["top_chunk_ids"], start=1):
            chunk_scores[int(chunk_id)] += community_score / rank

    for chunk_id, score in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[: args.candidates]:
        chunk_scores[chunk_id] += score * 0.2

    normalized_graph = minmax(dict(chunk_scores))
    normalized_bm25 = minmax(bm25_scores)
    results = []
    for chunk_id in set(normalized_graph) | set(normalized_bm25):
        chunk = chunks_by_id[chunk_id]
        graph = normalized_graph.get(chunk_id, 0.0)
        keyword = normalized_bm25.get(chunk_id, 0.0)
        phrase = text_overlap_score(args.question, chunk["text"])
        score = args.graph_weight * graph + args.keyword_weight * keyword + 0.25 * phrase
        results.append((score, chunk, {"community": graph, "keyword": keyword, "phrase": phrase}))
    results.sort(key=lambda item: item[0], reverse=True)
    return results[: args.top_k], selected_communities


def best_sentences(query: str, results: list[tuple[float, dict, dict[str, float]]], max_sentences: int) -> list[str]:
    query_terms = set(normalized_tokens(query))
    selected: list[str] = []
    seen: set[str] = set()

    for _, chunk, _ in results:
        repaired_text = repair_vietnamese_ocr_text(chunk["text"])
        sentences = [clean_text(part) for part in SENTENCE_RE.split(repaired_text) if clean_text(part)]
        if not sentences:
            sentences = [repaired_text]

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


def format_citation(chunk: dict) -> str:
    return f"{chunk['document']} p.{chunk['page']} chunk {chunk['chunk_on_page']}"


def graph_context(
    index: dict,
    mode: str,
    results: list[tuple[float, dict, dict[str, float]]],
    communities: list[dict],
    max_chars: int,
) -> str:
    blocks: list[str] = []
    used = 0

    if mode == "global" and communities:
        for community in communities:
            block = f"[community {community['id']}] {community['summary']}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)

    for rank, (_, chunk, _) in enumerate(results, start=1):
        entity_names = [
            index["entities"][key]["name"]
            for key in chunk.get("entities", [])[:8]
            if key in index["entities"]
        ]
        block = (
            f"[{rank}] {format_citation(chunk)}\n"
            f"Entities: {', '.join(entity_names)}\n"
            f"{repair_vietnamese_ocr_text(chunk['text'])}"
        )
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


def call_openai_llm(question: str, context: str, model: str, max_output_tokens: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    prompt = f"""You are a Vietnamese legal/regulatory GraphRAG assistant.

Answer the user's question using only the graph context below.
Write a natural, polished Vietnamese answer.
Prefer a short direct answer first, then bullet points when there are multiple rules or conditions.
Cite sources inline using bracket numbers like [1] or [2] for chunk evidence.
Community summaries can guide synthesis, but factual claims must come from cited chunks.
The retrieved text may contain OCR mistakes. You may silently correct obvious Vietnamese OCR noise, but do not change legal meaning.
If the context does not contain enough information, say: "Không tìm thấy đủ thông tin trong tài liệu được cung cấp."
Do not invent facts outside the context.

Question:
{question}

Graph context:
{context}
"""

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(
            {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}
        ).encode("utf-8"),
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


def should_use_llm(llm_mode: str) -> bool:
    if llm_mode == "openai":
        return True
    if llm_mode == "auto":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return False


def ask_graph(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index_path))
    mode = choose_mode(args.mode, args.question)

    if mode == "global":
        results, communities = global_retrieve(args, index)
        seed_names: list[str] = []
    else:
        results, seed_names = local_retrieve(args, index)
        communities = []

    if not results:
        print("No matching graph context found.")
        return

    print(f"\nGraph search mode: {mode}")
    if seed_names:
        print("Seed entities: " + ", ".join(seed_names))
    if communities:
        print("Communities: " + ", ".join(f"#{community['id']}" for community in communities))

    if should_use_llm(args.llm):
        print("\nNatural language answer")
        print("-----------------------")
        try:
            context = graph_context(index, mode, results, communities, args.llm_context_chars)
            print(call_openai_llm(args.question, context, args.llm_model, args.llm_max_output_tokens))
        except RuntimeError as exc:
            print(f"LLM error: {exc}")
            print("\nGraph extractive answer")
            print("-----------------------")
            for sentence in best_sentences(args.question, results, args.max_sentences):
                print(f"- {sentence}")
    else:
        print("\nGraph extractive answer")
        print("-----------------------")
        if args.llm == "auto":
            print("(Set OPENAI_API_KEY or pass --llm openai for a natural language answer.)")
        for sentence in best_sentences(args.question, results, args.max_sentences):
            print(f"- {sentence}")

    print("\nSources")
    print("-------")
    for rank, (score, chunk, components) in enumerate(results, start=1):
        preview = repair_vietnamese_ocr_text(chunk["text"])[: args.preview_chars]
        detail = " ".join(f"{key}={value:.3f}" for key, value in components.items())
        entity_names = [
            index["entities"][key]["name"]
            for key in chunk.get("entities", [])[:5]
            if key in index["entities"]
        ]
        print(f"[{rank}] score={score:.3f} {detail} {format_citation(chunk)}")
        if entity_names:
            print(f"    entities: {', '.join(entity_names)}")
        print(f"    {preview}")


def inspect_graph(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index_path))
    stats = index["stats"]
    print(f"Source: {index.get('source', 'unknown')}")
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Entities: {stats['entities']}")
    print(f"Relationships: {stats['relationships']}")
    print(f"Communities: {stats['communities']}")
    print(f"Average chunk terms: {stats['avg_chunk_terms']:.1f}")
    print(f"Index path: {args.index_path}")

    print("\nTop communities")
    for community in index["communities"][:5]:
        print(f"- #{community['id']} entities={len(community['entities'])}: {', '.join(community['top_entities'][:8])}")


def add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Folder containing PDF files.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Output GraphRAG index path.")
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
    parser.add_argument("--ocr-lang", default=DEFAULT_OCR_LANG, help="Tesseract language code, for example vie+eng.")
    parser.add_argument("--ocr-dpi", type=int, default=DEFAULT_OCR_DPI, help="DPI used when rendering PDFs for OCR.")
    parser.add_argument("--ocr-psm", default=DEFAULT_OCR_PSM, help="Tesseract page segmentation mode.")
    parser.add_argument("--ocr-timeout", type=int, default=180, help="Timeout in seconds for OCR rendering/OCR.")
    parser.add_argument("--max-entities-per-chunk", type=int, default=18, help="Entity mentions kept per chunk.")
    parser.add_argument("--max-graph-entities-per-chunk", type=int, default=12, help="Entities used for co-occurrence edges.")
    parser.add_argument("--community-iterations", type=int, default=12, help="Weighted label propagation passes.")
    parser.add_argument("--community-top-chunks", type=int, default=8, help="Chunks retained in each community summary.")
    parser.add_argument("--max-edge-chunk-refs", type=int, default=10, help="Chunk ids retained per relationship.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone local GraphRAG over a PDF folder.", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a graph index from PDFs.")
    add_build_args(build_parser)
    build_parser.set_defaults(func=build_graph_index)

    ask_parser = subparsers.add_parser("ask", help="Ask using GraphRAG retrieval.")
    ask_parser.add_argument("question", help="Question to search for.")
    ask_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to GraphRAG index.")
    ask_parser.add_argument("--mode", choices=["auto", "local", "global"], default="auto", help="Graph search mode.")
    ask_parser.add_argument("--candidates", type=int, default=40, help="Candidate chunks before final ranking.")
    ask_parser.add_argument("--top-k", type=int, default=7, help="Final chunks returned.")
    ask_parser.add_argument("--seed-entities", type=int, default=8, help="Seed entities for local graph search.")
    ask_parser.add_argument("--neighbors", type=int, default=8, help="Neighbors expanded for each seed entity.")
    ask_parser.add_argument("--communities", type=int, default=5, help="Communities used for global graph search.")
    ask_parser.add_argument("--graph-weight", type=float, default=0.60, help="Graph score contribution.")
    ask_parser.add_argument("--keyword-weight", type=float, default=0.40, help="BM25 score contribution.")
    ask_parser.add_argument("--max-sentences", type=int, default=6, help="Number of extractive answer sentences.")
    ask_parser.add_argument("--preview-chars", type=int, default=420, help="Characters shown per source.")
    ask_parser.add_argument(
        "--llm",
        choices=["auto", "none", "openai"],
        default="auto",
        help="Use OpenAI for a natural answer. auto uses it when OPENAI_API_KEY is set.",
    )
    ask_parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    ask_parser.add_argument("--llm-max-output-tokens", type=int, default=900)
    ask_parser.add_argument("--llm-context-chars", type=int, default=7000)
    ask_parser.set_defaults(func=ask_graph)

    inspect_parser = subparsers.add_parser("inspect", help="Show GraphRAG index statistics.")
    inspect_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    inspect_parser.set_defaults(func=inspect_graph)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
