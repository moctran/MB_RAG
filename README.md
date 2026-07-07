# Naive PDF RAG

This is a small local RAG baseline over the PDF files in `pdf_files/`.

It does three things:

1. Extracts text from each PDF.
2. Splits each page into overlapping chunks.
3. Retrieves relevant chunks with BM25 and returns an extractive answer with citations.

## Setup

The existing `rag_index.json` and OCR-only builds do not require Python packages.
Create and activate a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On macOS/Homebrew Python, installing into the system interpreter may fail with
`externally-managed-environment`. Use the virtual environment commands above
instead of `--break-system-packages`.

The packages in `requirements.txt` are only needed for non-OCR PDF extraction
(`pypdfium2`, `pdfplumber`, `pypdf`). If the network is slow, skip the pip
upgrade and install with a longer timeout:

```bash
python3 -m pip install --timeout 120 --retries 10 -r requirements.txt
```

## Build The Index

```bash
python3 naive_rag.py build
```

This reads every `*.pdf` in `pdf_files/` and writes `rag_index.json`.

Useful options:

```bash
python3 naive_rag.py build --chunk-size 450 --overlap 80
python3 naive_rag.py build --pdf-dir pdf_files --index-path rag_index.json
python3 naive_rag.py build --extractor pdfplumber --extract-timeout 30
python3 naive_rag.py build --extractor ocr --ocr-lang eng --ocr-dpi 120
```

## Ask Questions

```bash
python3 naive_rag.py ask "Quy định về hệ thống kiểm soát nội bộ ngân hàng là gì?"
python3 naive_rag.py ask "Thông tư 09/2020 quy định gì về an toàn hệ thống thông tin?"
```

The answer is extractive: it quotes or condenses the most relevant retrieved text. The `Sources` section shows the PDF, page, chunk number, BM25 score, and a preview of each retrieved chunk.

## Ask With An LLM

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Then add `--llm openai`:

```bash
python3 naive_rag.py ask "bảo vệ dữ liệu cá nhân phạm vi điều chỉnh" --llm openai
```

Useful options:

```bash
python3 naive_rag.py ask "câu hỏi của bạn" --llm openai --llm-model gpt-5.5
python3 naive_rag.py ask "câu hỏi của bạn" --llm openai --top-k 8 --llm-context-chars 1800
```

The LLM is instructed to answer only from retrieved PDF chunks and cite sources like `[1]`, `[2]`. If the API key is missing or the API call fails, the script falls back to the extractive answer.

## Inspect The Index

```bash
python3 naive_rag.py inspect
```

## Advanced RAG

The advanced version is independent from `naive_rag.py`. It reads PDFs directly
from `pdf_files/` and writes its own index under `advanced_index/`.

```bash
python3 advanced_rag.py build --extractor ocr --ocr-lang vie+eng --ocr-dpi 180
```

Default advanced index path:

```text
advanced_index/advanced_rag_index.json
```

Ask with hybrid retrieval:

```bash
python3 advanced_rag.py ask "bảo vệ dữ liệu cá nhân phạm vi điều chỉnh"
```

Ask with hybrid retrieval plus OpenAI LLM generation:

```bash
export OPENAI_API_KEY="your_api_key_here"
python3 advanced_rag.py ask "bảo vệ dữ liệu cá nhân phạm vi điều chỉnh" --llm openai
```

Implemented advanced features:

- Independent PDF extraction and indexing, separate from `rag_index.json`.
- Query rewriting: adds accent-insensitive Vietnamese variants and domain expansions.
- Hybrid search: combines normalized BM25 keyword retrieval with local hashed TF-IDF semantic vectors.
- Re-ranking: adjusts retrieved chunks using query-term coverage and density.
- Context compression: keeps the most relevant sentences before answer generation.
- Optional LLM answer: sends the compressed retrieved context to OpenAI and keeps source citations.

Not implemented in this local version:

- MyScaleDB MSTG indexing. That requires running MyScaleDB or another vector database.
- True neural embeddings or fine-tuned dynamic embeddings. The current version uses local hashed TF-IDF vectors so it works without model downloads.
- Neural cross-encoder re-ranking. The current re-ranker is lexical/semantic-score based.

## FAISS + OpenAI RAG

The FAISS version is separate from both `naive_rag.py` and `advanced_rag.py`.
It embeds PDF chunks with OpenAI embeddings and stores vectors in:

```text
faiss_index/index.faiss
faiss_index/metadata.json
```

Install dependencies:

```bash
source .venv/bin/activate
python3 -m pip install --timeout 120 --retries 10 -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Build the FAISS index:

```bash
python3 faiss_rag.py build --extractor ocr --ocr-lang vie+eng --ocr-dpi 180
```

Ask with LLM generation:

```bash
python3 faiss_rag.py ask "bảo vệ dữ liệu cá nhân phạm vi điều chỉnh" --llm openai
```

Inspect retrieval without the LLM:

```bash
python3 faiss_rag.py ask "bảo vệ dữ liệu cá nhân phạm vi điều chỉnh" --no-llm
```

Inspect the index:

```bash
python3 faiss_rag.py inspect
```

Notes:

- `build` requires `OPENAI_API_KEY` because embeddings are created through the OpenAI Embeddings API.
- `ask` also requires `OPENAI_API_KEY` because the question is embedded before FAISS search.
- The script calls OpenAI with Python's standard library, so the `openai` Python package is not required.
- The default embedding model is `text-embedding-3-small`.
- The FAISS index uses normalized vectors with `IndexFlatIP`, which gives cosine-similarity style search.

## Notes

- This version does not call an LLM. It is the retrieval baseline you can use before adding embeddings, a vector database, or a generator model.
- Extraction defaults to `--extractor auto`, which tries Poppler `pdftotext` first, then `pypdfium2`, `pdfplumber`, `pypdf`, and OCR.
- Each extraction attempt has a timeout, so a malformed PDF will be skipped instead of blocking the whole build.
- OCR requires `pdftoppm` and `tesseract`. This machine has English OCR data installed; install Vietnamese Tesseract data and run `--ocr-lang vie+eng` for better Vietnamese output.
- `pypdf` is kept as a fallback, but PDFium/`pdfplumber` are usually better defaults for this corpus.
