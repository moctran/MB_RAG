#!/usr/bin/env python3
"""Ontology-constrained RAG for Vietnamese banking and legal PDFs.

The ontology is intentionally editable and small.  It constrains graph nodes and
predicates while the original text remains available for BM25 retrieval and
page-level citations. PDF extraction is shared with graph_rag.py so all local
RAG variants use the same tested OCR fallbacks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from graph_rag import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANG,
    DEFAULT_OCR_PSM,
    clean_text,
    discover_pdfs,
    extract_pdf_pages,
    repair_vietnamese_ocr_text,
    strip_vietnamese_accents,
)


DEFAULT_PDF_DIR = Path("pdf_files")
DEFAULT_ONTOLOGY_PATH = Path("banking_ontology.json")
DEFAULT_INDEX_PATH = Path("ontology_index/ontology_rag_index.json")
DEFAULT_OPENAI_MODEL = "gpt-5.5"
WORD_RE = re.compile(r"(?u)\b\w+\b")
ARTICLE_RE = re.compile(r"(?iu)\bĐi[eề]u\s+(\d+[a-zđ]?)\s*[.:]?\s*")
LEGAL_DOC_RE = re.compile(
    r"(?iu)\b(?:s[oố]\s*[:.]?\s*)?"
    r"(\d{1,3}\s*(?:/\s*\d{4})?\s*/\s*(?:TT|NĐ|ND|QĐ|QD|NQ)\s*-\s*[A-ZĐ-]+)\b"
)
DATE_RE = re.compile(r"(?iu)ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})")
TYPE_PATTERNS = [
    ("Circular", re.compile(r"(?iu)\bth[oô]ng\s+t[ưứ]\b|/TT-")),
    ("Decree", re.compile(r"(?iu)\bngh[iị]\s+[đd]ịnh\b|/(?:NĐ|ND)-")),
    ("Decision", re.compile(r"(?iu)\bquy[eế]t\s+[đd]ịnh\b|/(?:QĐ|QD)-")),
    ("Resolution", re.compile(r"(?iu)\bngh[iị]\s+quy[eế]t\b|/NQ-")),
    ("Law", re.compile(r"(?iu)\blu[aậ]t\b")),
]
PREDICATE_PATTERNS = {
    "requires": re.compile(r"(?iu)\b(ph[aả]i|c[oó]\s+tr[aá]ch\s+nhi[eệ]m|b[aắ]t\s+bu[oộ]c)\b"),
    "prohibits": re.compile(r"(?iu)\b(kh[oô]ng\s+[đd][ưượ]c|nghi[eê]m\s+c[aấ]m|b[iị]\s+c[aấ]m)\b"),
    "permits": re.compile(r"(?iu)\b([đd][ưượ]c\s+ph[eé]p|cho\s+ph[eé]p)\b"),
    "has_right": re.compile(r"(?iu)\b(c[oó]\s+quy[eề]n|quy[eề]n\s+[đd][ưượ]c)\b"),
    "amends": re.compile(r"(?iu)\b(s[ưử]a\s+[đd][oổ]i|b[oổ]\s+sung)\b"),
    "replaces": re.compile(r"(?iu)\b(thay\s+th[eế]|h[eế]t\s+hi[eệ]u\s+l[ưự]c)\b"),
}
SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def normalized(text: str) -> str:
    unaccented = strip_vietnamese_accents(text.lower())
    return " ".join(WORD_RE.findall(unaccented))


def tokens(text: str) -> list[str]:
    return [normalized(token) for token in WORD_RE.findall(text) if len(token) > 1]


def text_quality_score(text: str) -> float:
    nonspace = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in nonspace) / max(1, len(nonspace))
    words = WORD_RE.findall(text)
    plausible = sum(len(word) >= 3 and sum(char.isalpha() for char in word) / len(word) >= 0.8 for word in words)
    plausible_ratio = plausible / max(1, len(words))
    return round(max(0.0, min(1.0, 0.55 * alpha_ratio + 0.45 * min(1.0, plausible_ratio / 0.62))), 3)


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {label} {path}: {exc}") from exc


def validate_ontology(ontology: dict) -> None:
    required = {"entity_types", "predicates", "concepts", "query_intents"}
    missing = sorted(required - set(ontology))
    if missing:
        raise SystemExit("Ontology is missing keys: " + ", ".join(missing))
    ids = [item.get("id") for item in ontology["concepts"]]
    if None in ids or len(ids) != len(set(ids)):
        raise SystemExit("Every ontology concept needs a unique id.")
    known_types = {child for children in ontology["entity_types"].values() for child in children}
    invalid_types = sorted({item["type"] for item in ontology["concepts"]} - known_types)
    if invalid_types:
        raise SystemExit("Unknown concept types: " + ", ".join(invalid_types))


def concept_catalog(ontology: dict) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    concepts = {item["id"]: item for item in ontology["concepts"]}
    aliases: list[tuple[str, str]] = []
    for item in ontology["concepts"]:
        for alias in set(item.get("aliases", []) + [item["label"]]):
            aliases.append((normalized(alias), item["id"]))
    aliases.sort(key=lambda pair: (-len(pair[0]), pair[0]))
    return concepts, aliases


def find_concepts(text: str, aliases: list[tuple[str, str]]) -> list[str]:
    haystack = f" {normalized(text)} "
    found: list[str] = []
    for alias, concept_id in aliases:
        if f" {alias} " in haystack and concept_id not in found:
            found.append(concept_id)
    return found


def legal_document_refs(text: str) -> list[str]:
    refs: list[str] = []
    for match in LEGAL_DOC_RE.finditer(strip_vietnamese_accents(text).upper()):
        value = re.sub(r"\s+", "", match.group(1)).replace("ND-", "NĐ-").replace("QD-", "QĐ-")
        if value not in refs:
            refs.append(value)
    return refs


def filename_document_ref(path: Path) -> str | None:
    name = strip_vietnamese_accents(path.stem).upper()
    patterns = [
        r"(?<!\d)(\d{1,3})[._-](\d{4})[._-](TT|ND|QD|NQ)[._-](NHNN|CP)(?![A-Z])",
        r"(?<!\d)(\d{1,3})-(\d{4})-(TT|ND|QD|NQ)-(NHNN|CP)(?![A-Z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            number, year, kind, issuer = match.groups()
            kind = {"ND": "NĐ", "QD": "QĐ"}.get(kind, kind)
            return f"{number}/{year}/{kind}-{issuer}"
    return None


def document_metadata(path: Path, pages: list[tuple[int, str]], concepts: list[str]) -> dict:
    leading = " ".join(text for _, text in pages[:3])[:10000]
    searchable = f"{path.stem} {leading}"
    header = re.split(r"(?iu)\bcăn\s+cứ\b", leading, maxsplit=1)[0]
    refs = legal_document_refs(header)
    primary_ref = filename_document_ref(path) or (refs[0] if refs else None)
    doc_type = "OtherLegalDocument"
    for candidate, pattern in TYPE_PATTERNS:
        if pattern.search(searchable):
            doc_type = candidate
            break
    dates = []
    for day, month, year in DATE_RE.findall(leading[:5000]):
        try:
            dates.append(datetime(int(year), int(month), int(day)).date().isoformat())
        except ValueError:
            pass
    all_text = " ".join(text for _, text in pages)
    words = WORD_RE.findall(all_text)
    suspicious = sum(
        1 for word in words
        if len(word) > 2 and (sum(ch.isalpha() for ch in word) / len(word) < 0.55 or re.search(r"\d.*[A-Za-z].*\d", word))
    )
    text_pages = sum(1 for _, text in pages if len(text.strip()) >= 40)
    coverage = text_pages / max(1, len(pages))
    noise_ratio = suspicious / max(1, len(words))
    quality_score = max(0.0, min(1.0, coverage * (1.0 - min(0.8, noise_ratio * 4))))
    status = "good" if quality_score >= 0.75 else "review" if quality_score >= 0.45 else "poor"
    return {
        "id": f"document:{path.name}",
        "source": str(path),
        "filename": path.name,
        "document_number": primary_ref,
        "document_type": doc_type,
        "issuer": next((cid for cid in ("state_bank", "government", "ministry_public_security") if cid in concepts), None),
        "promulgation_date": dates[0] if dates else None,
        "pages": len(pages),
        "text_pages": text_pages,
        "quality_score": round(quality_score, 3),
        "quality_status": status,
        "concepts": concepts,
    }


def article_segments(page_text: str, current_article: str | None) -> tuple[list[tuple[str | None, str]], str | None]:
    matches = list(ARTICLE_RE.finditer(page_text))
    if not matches:
        return [(current_article, page_text)], current_article
    segments: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        segments.append((current_article, page_text[: matches[0].start()]))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page_text)
        article = match.group(1)
        segments.append((article, page_text[match.start():end]))
        current_article = article
    return segments, current_article


def word_chunks(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    result = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        result.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return result


def merge_overlapping_texts(texts: list[str]) -> str:
    """Reconstruct a page from cached overlapping word chunks."""
    if not texts:
        return ""
    merged = texts[0].split()
    for text in texts[1:]:
        incoming = text.split()
        max_overlap = min(len(merged), len(incoming), 160)
        overlap = 0
        for width in range(max_overlap, 9, -1):
            if merged[-width:] == incoming[:width]:
                overlap = width
                break
        merged.extend(incoming[overlap:])
    return " ".join(merged)


def cached_pages_by_document(path: Path) -> dict[str, list[tuple[int, str]]]:
    cached = load_json(path, "Reusable graph index")
    grouped: dict[str, dict[int, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for chunk in cached.get("chunks", []):
        document = chunk.get("document") or Path(chunk["source"]).name
        grouped[document][int(chunk["page"])].append((int(chunk.get("chunk_on_page", 1)), chunk["text"]))
    result: dict[str, list[tuple[int, str]]] = {}
    for document, page_map in grouped.items():
        result[document] = [
            (page, merge_overlapping_texts([text for _, text in sorted(items)]))
            for page, items in sorted(page_map.items())
        ]
    return result


def sentence_facts(
    text: str,
    document_id: str,
    article: str | None,
    chunk_id: int,
    concept_ids: list[str],
    refs: list[str],
    concepts: dict[str, dict],
    aliases: list[tuple[str, str]],
) -> list[dict]:
    facts: list[dict] = []
    subject_default = f"{document_id}#article:{article}" if article else document_id
    sentences = [clean_text(part) for part in SENTENCE_RE.split(text) if len(clean_text(part)) >= 20]
    for sentence_index, sentence in enumerate(sentences):
        sentence_concepts = find_concepts(sentence, aliases)
        sentence_refs = [ref for ref in refs if normalized(ref) in normalized(sentence)]
        predicates = [name for name, pattern in PREDICATE_PATTERNS.items() if pattern.search(sentence)]
        if "doi tuong ap dung" in normalized(sentence):
            predicates.append("applies_to")
        if "giai thich tu ngu" in normalized(text) and re.search(r"(?iu)\b(l[aà]|được\s+hiểu\s+là)\b", sentence):
            predicates.append("defines")
        if sentence_refs:
            predicates.append("references")
        if not predicates and sentence_concepts:
            predicates.append("mentions")
        actor_ids = [cid for cid in sentence_concepts if concepts[cid]["type"] in {"Authority", "FinancialInstitution", "Organization", "Person", "DataRole"}]
        topic_ids = [cid for cid in sentence_concepts if cid not in actor_ids]
        for predicate in dict.fromkeys(predicates):
            if predicate == "has_right" and actor_ids:
                subject = actor_ids[0]
            elif predicate == "requires" and actor_ids:
                subject = actor_ids[0]
            else:
                subject = subject_default
            targets = topic_ids or actor_ids or sentence_refs
            for target in targets[:5]:
                if target == subject:
                    continue
                facts.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": f"legal_document:{target}" if target in sentence_refs else target,
                    "chunk_id": chunk_id,
                    "sentence": sentence[:700],
                    "sentence_index": sentence_index,
                    "confidence": 0.9 if predicate != "mentions" else 0.65,
                })
    return facts


def build_index(args: argparse.Namespace) -> None:
    ontology_path = Path(args.ontology_path)
    ontology = load_json(ontology_path, "Ontology")
    validate_ontology(ontology)
    concepts, aliases = concept_catalog(ontology)
    pdfs = discover_pdfs(Path(args.pdf_dir))
    documents: list[dict] = []
    chunks: list[dict] = []
    facts: list[dict] = []
    cached_pages = cached_pages_by_document(Path(args.reuse_graph_index)) if args.reuse_graph_index else {}

    for pdf_path in pdfs:
        print(f"Reading {pdf_path.name} ...", flush=True)
        pages = cached_pages.get(pdf_path.name) or extract_pdf_pages(args, pdf_path)
        full_text = " ".join(text for _, text in pages)
        doc_concepts = find_concepts(full_text, aliases)
        metadata = document_metadata(pdf_path, pages, doc_concepts)
        current_article = None
        doc_chunk_count = 0
        for page, raw_text in pages:
            page_text = repair_vietnamese_ocr_text(raw_text)
            if len(page_text) < args.min_page_chars:
                continue
            segments, current_article = article_segments(page_text, current_article)
            for article, segment in segments:
                if len(segment.strip()) < 20:
                    continue
                for text in word_chunks(segment, args.chunk_size, args.overlap):
                    term_counts = Counter(tokens(text))
                    if not term_counts:
                        continue
                    chunk_id = len(chunks)
                    concept_ids = find_concepts(text, aliases)
                    refs = legal_document_refs(text)
                    chunk = {
                        "id": chunk_id,
                        "document_id": metadata["id"],
                        "document": pdf_path.name,
                        "source": str(pdf_path),
                        "page": page,
                        "article": article,
                        "text": text,
                        "term_counts": dict(term_counts),
                        "length": sum(term_counts.values()),
                        "quality_score": text_quality_score(text),
                        "concepts": concept_ids,
                        "legal_references": refs,
                    }
                    chunks.append(chunk)
                    facts.extend(sentence_facts(text, metadata["id"], article, chunk_id, concept_ids, refs, concepts, aliases))
                    doc_chunk_count += 1
        metadata["chunks"] = doc_chunk_count
        documents.append(metadata)
        print(
            f"  pages={metadata['pages']} chunks={doc_chunk_count} quality={metadata['quality_status']} "
            f"({metadata['quality_score']:.2f}) concepts={len(doc_concepts)}"
            + (" source=cached-ocr" if pdf_path.name in cached_pages else " source=pdf"),
            flush=True,
        )

    if not chunks:
        raise SystemExit("No chunks created. Try --extractor ocr with the correct Tesseract languages.")
    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(chunk["term_counts"])
    avg_length = sum(chunk["length"] for chunk in chunks) / len(chunks)
    predicate_counts = Counter(fact["predicate"] for fact in facts)
    index = {
        "version": 1,
        "source": "ontology_rag",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "ontology_path": str(ontology_path),
        "ontology": ontology,
        "settings": {key: value for key, value in vars(args).items() if key != "func"},
        "stats": {
            "documents": len(documents),
            "chunks": len(chunks),
            "facts": len(facts),
            "avg_chunk_terms": avg_length,
            "quality_review_documents": sum(doc["quality_status"] != "good" for doc in documents),
            "predicates": dict(predicate_counts),
        },
        "documents": documents,
        "doc_freq": dict(doc_freq),
        "chunks": chunks,
        "facts": facts,
    }
    output = Path(args.index_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote ontology index to {output}")
    print(f"Documents={len(documents)} chunks={len(chunks)} typed_facts={len(facts)}")


def bm25_scores(index: dict, query: str) -> dict[int, float]:
    query_terms = Counter(tokens(query))
    total = len(index["chunks"])
    avgdl = max(1.0, float(index["stats"]["avg_chunk_terms"]))
    scores: dict[int, float] = {}
    for chunk in index["chunks"]:
        score = 0.0
        for term, count in query_terms.items():
            tf = int(chunk["term_counts"].get(term, 0))
            if not tf:
                continue
            df = int(index["doc_freq"].get(term, 0))
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            score += count * idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * chunk["length"] / avgdl))
        if score:
            scores[int(chunk["id"])] = score
    return scores


def minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def query_semantics(index: dict, question: str) -> tuple[list[str], list[str], list[str]]:
    concepts, aliases = concept_catalog(index["ontology"])
    del concepts
    matched_concepts = find_concepts(question, aliases)
    query_norm = normalized(question)
    predicates = []
    for predicate, hints in index["ontology"]["query_intents"].items():
        if any(normalized(hint) in query_norm for hint in hints):
            predicates.append(predicate)
    refs = legal_document_refs(question)
    return matched_concepts, predicates, refs


def retrieve(index: dict, question: str, top_k: int) -> tuple[list[tuple[float, dict, dict]], dict]:
    query_concepts, predicates, refs = query_semantics(index, question)
    keyword = minmax(bm25_scores(index, question))
    ontology_raw: Counter[int] = Counter()
    documents = {doc["id"]: doc for doc in index["documents"]}
    for chunk in index["chunks"]:
        shared = set(query_concepts) & set(chunk.get("concepts", []))
        if shared:
            ontology_raw[int(chunk["id"])] += 1.5 * len(shared)
        if refs and set(refs) & set(chunk.get("legal_references", [])):
            ontology_raw[int(chunk["id"])] += 2.0
        document = documents.get(chunk["document_id"], {})
        if refs and document.get("document_number") in refs:
            ontology_raw[int(chunk["id"])] += 3.0
    for fact in index["facts"]:
        concept_hit = fact["subject"] in query_concepts or fact["object"] in query_concepts
        predicate_hit = fact["predicate"] in predicates
        ref_hit = any(ref in fact["object"] for ref in refs)
        if concept_hit:
            ontology_raw[int(fact["chunk_id"])] += 1.0
        if predicate_hit:
            ontology_raw[int(fact["chunk_id"])] += 1.25
        if concept_hit and predicate_hit:
            ontology_raw[int(fact["chunk_id"])] += 1.5
        if ref_hit:
            ontology_raw[int(fact["chunk_id"])] += 2.0
    ontology_scores = minmax(dict(ontology_raw))
    candidates = set(keyword) | set(ontology_scores)
    matching_documents = {
        doc["id"] for doc in index["documents"]
        if doc.get("document_number") in refs
    }
    if matching_documents and "references" not in predicates:
        candidates = {
            chunk_id for chunk_id in candidates
            if index["chunks"][chunk_id]["document_id"] in matching_documents
        }
    ranked = []
    for chunk_id in candidates:
        kw = keyword.get(chunk_id, 0.0)
        ont = ontology_scores.get(chunk_id, 0.0)
        chunk = index["chunks"][chunk_id]
        article_bonus = 0.08 if chunk.get("article") and re.search(rf"(?iu)\bđi[eề]u\s+{re.escape(str(chunk['article']))}\b", question) else 0.0
        intent_bonus = 0.0
        opening = normalized(chunk["text"][:180])
        if "has_right" in predicates and "quyen cua" in opening:
            intent_bonus = 0.28
        elif "applies_to" in predicates and "doi tuong ap dung" in opening:
            intent_bonus = 0.16
        semantic_gate = 0.25 if (query_concepts or refs) and ont == 0 else 1.0
        quality = float(chunk.get("quality_score", text_quality_score(chunk["text"])))
        quality_factor = 0.2 + 0.8 * quality
        score = (argsafe_weight(kw, 0.55) * semantic_gate + argsafe_weight(ont, 0.45) + article_bonus + intent_bonus) * quality_factor
        ranked.append((score, chunk, {"keyword": kw, "ontology": ont, "quality": quality, "article": article_bonus, "intent": intent_bonus}))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    semantics = {"concepts": query_concepts, "predicates": predicates, "legal_references": refs}
    return ranked[:top_k], semantics


def argsafe_weight(value: float, weight: float) -> float:
    return value * weight


def best_sentences(question: str, results: list[tuple[float, dict, dict]], count: int) -> list[tuple[str, int]]:
    query_terms = set(tokens(question))
    query_refs = set(legal_document_refs(question))
    candidates = []
    seen = set()
    for rank, (_, chunk, _) in enumerate(results, start=1):
        for sentence in SENTENCE_RE.split(repair_vietnamese_ocr_text(chunk["text"])):
            sentence = clean_text(sentence)
            if len(sentence) < 35:
                continue
            sentence_refs = set(legal_document_refs(sentence))
            if query_refs and sentence_refs and not (query_refs & sentence_refs):
                continue
            key = normalized(sentence)
            if key in seen:
                continue
            seen.add(key)
            sentence_terms = set(tokens(sentence))
            coverage = len(query_terms & sentence_terms) / max(1, len(query_terms))
            candidates.append((coverage + 0.15 / rank, sentence, rank))
    return [(sentence, rank) for _, sentence, rank in sorted(candidates, reverse=True)[:count]]


def context_for_llm(index: dict, results: list[tuple[float, dict, dict]], limit: int) -> str:
    concepts = {item["id"]: item for item in index["ontology"]["concepts"]}
    documents = {item["id"]: item for item in index["documents"]}
    parts = []
    for rank, (_, chunk, _) in enumerate(results, start=1):
        labels = [concepts[cid]["label"] for cid in chunk.get("concepts", []) if cid in concepts]
        article = f", Điều {chunk['article']}" if chunk.get("article") else ""
        part = f"[{rank}] {chunk['document']}, trang {chunk['page']}{article}"
        document_number = documents.get(chunk["document_id"], {}).get("document_number")
        if document_number:
            part += f"\nVerified document number from manifest: {document_number}"
        if labels:
            part += "\nOntology concepts: " + ", ".join(labels)
        part += "\n" + repair_vietnamese_ocr_text(chunk["text"])
        parts.append(part)
    return "\n\n".join(parts)[:limit]


def call_openai(question: str, context: str, model: str, max_tokens: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    prompt = f"""Bạn là trợ lý tra cứu văn bản pháp luật Việt Nam.
Chỉ trả lời từ bằng chứng được cung cấp. Trích dẫn nguồn bằng [1], [2].
Phân biệt rõ yêu cầu, quyền, điều cấm và ngoại lệ. Nếu bằng chứng không đủ, hãy nói rõ.

Câu hỏi: {question}

Bằng chứng:
{context}
"""
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps({"model": model, "input": prompt, "max_output_tokens": max_tokens}).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        detail = exc.read().decode(errors="replace") if isinstance(exc, urllib.error.HTTPError) else str(exc)
        raise RuntimeError(f"OpenAI request failed: {detail}") from exc
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise RuntimeError("OpenAI returned no text")


def ask(args: argparse.Namespace) -> None:
    index = load_json(Path(args.index_path), "Index")
    results, semantics = retrieve(index, args.question, args.top_k)
    if not results:
        print("No matching evidence found.")
        return
    concepts = {item["id"]: item for item in index["ontology"]["concepts"]}
    documents = {item["id"]: item for item in index["documents"]}
    print("\nOntology interpretation")
    print("-----------------------")
    print("Concepts: " + (", ".join(concepts[cid]["label"] for cid in semantics["concepts"]) or "none"))
    print("Predicates: " + (", ".join(semantics["predicates"]) or "none"))
    print("Legal references: " + (", ".join(semantics["legal_references"]) or "none"))
    use_llm = args.llm == "openai" or (args.llm == "auto" and os.environ.get("OPENAI_API_KEY"))
    print("\nAnswer")
    print("------")
    if use_llm:
        try:
            print(call_openai(args.question, context_for_llm(index, results, args.llm_context_chars), args.llm_model, args.llm_max_output_tokens))
        except RuntimeError as exc:
            print(f"LLM error: {exc}\nFalling back to relevant extracts:")
            for sentence, rank in best_sentences(args.question, results, args.max_sentences):
                print(f"- {sentence} [{rank}]")
    else:
        for sentence, rank in best_sentences(args.question, results, args.max_sentences):
            print(f"- {sentence} [{rank}]")
    print("\nSources")
    print("-------")
    for rank, (score, chunk, components) in enumerate(results, start=1):
        article = f", Điều {chunk['article']}" if chunk.get("article") else ""
        labels = [concepts[cid]["label"] for cid in chunk.get("concepts", []) if cid in concepts]
        document_number = documents.get(chunk["document_id"], {}).get("document_number")
        print(f"[{rank}] score={score:.3f} keyword={components['keyword']:.3f} ontology={components['ontology']:.3f} quality={components['quality']:.2f} "
              f"{chunk['document']}, trang {chunk['page']}{article}")
        if document_number:
            print(f"    document number: {document_number}")
        if labels:
            print("    concepts: " + ", ".join(labels))
        print("    " + repair_vietnamese_ocr_text(chunk["text"])[:args.preview_chars])


def inspect_index(args: argparse.Namespace) -> None:
    index = load_json(Path(args.index_path), "Index")
    stats = index["stats"]
    print(f"Source: {index.get('source')}")
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Typed facts: {stats['facts']}")
    print(f"Documents needing quality review: {stats['quality_review_documents']}")
    print("Predicates: " + ", ".join(f"{key}={value}" for key, value in sorted(stats["predicates"].items())))
    print("\nDocument manifest")
    for doc in index["documents"]:
        number = doc["document_number"] or "unknown number"
        print(f"- [{doc['quality_status']}] {doc['filename']}: {doc['document_type']} {number}, "
              f"quality={doc['quality_score']:.2f}, chunks={doc['chunks']}, concepts={len(doc['concepts'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ontology-constrained RAG over Vietnamese banking PDFs.", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build the ontology index.")
    build.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    build.add_argument("--ontology-path", default=str(DEFAULT_ONTOLOGY_PATH))
    build.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    build.add_argument(
        "--reuse-graph-index",
        help="Reuse page text from an existing graph index to avoid repeating slow OCR.",
    )
    build.add_argument("--chunk-size", type=int, default=320)
    build.add_argument("--overlap", type=int, default=50)
    build.add_argument("--min-page-chars", type=int, default=40)
    build.add_argument("--extractor", choices=["auto", "pdftotext", "pypdfium2", "pypdf", "pdfplumber", "ocr"], default="auto")
    build.add_argument("--extract-timeout", type=int, default=15)
    build.add_argument("--ocr-lang", default=DEFAULT_OCR_LANG)
    build.add_argument("--ocr-dpi", type=int, default=DEFAULT_OCR_DPI)
    build.add_argument("--ocr-psm", default=DEFAULT_OCR_PSM)
    build.add_argument("--ocr-timeout", type=int, default=180)
    build.set_defaults(func=build_index)
    ask_parser = sub.add_parser("ask", help="Ask with hybrid keyword + ontology retrieval.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    ask_parser.add_argument("--top-k", type=int, default=7)
    ask_parser.add_argument("--max-sentences", type=int, default=6)
    ask_parser.add_argument("--preview-chars", type=int, default=420)
    ask_parser.add_argument("--llm", choices=["auto", "none", "openai"], default="auto")
    ask_parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    ask_parser.add_argument("--llm-max-output-tokens", type=int, default=900)
    ask_parser.add_argument("--llm-context-chars", type=int, default=8000)
    ask_parser.set_defaults(func=ask)
    inspect_parser = sub.add_parser("inspect", help="Inspect ontology and document quality statistics.")
    inspect_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    inspect_parser.set_defaults(func=inspect_index)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
