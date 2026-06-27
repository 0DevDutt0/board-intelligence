# CLAUDE.md — Board Intelligence System
# Claude Code reads this file automatically on every session.
# Do not delete or rename this file.

## Project Identity

This is an **enterprise-grade, fully offline Board Intelligence System** built
with RAG (Retrieval-Augmented Generation). It ingests a confidential board
meeting PDF at session start and provides cited natural-language Q&A — entirely
on local hardware, zero network egress, zero external API calls.

## Hardware Context (Critical)

- **GPU**: NVIDIA RTX 5090 Laptop — Blackwell architecture, sm_120
- **VRAM**: 24,463 MiB (~24 GB)
- **Driver**: 592.01
- **CUDA Runtime**: 13.1 reported by nvidia-smi (this is the MAX supported,
  NOT the build target)
- **Build target**: CUDA 12.8 — mandatory for sm_120 MMQ kernel stability
- **WSL2**: 2.7.8.0 — required for CUDA graph capture on Blackwell
- **RAM**: 23 GB
- **Disk**: ~898 GB free
- **OS**: Windows 11 24H2 + WSL2 Ubuntu

### Why CUDA 12.8, not 13.x

CUDA 13.1 causes MMQ kernel segfaults on Blackwell (sm_120) in both vLLM and
llama.cpp. The correct build always targets CUDA 12.8 explicitly. The driver
supports 13.1 but that does NOT mean we compile against it.

### vLLM on Blackwell WSL2

- Requires WSL2 >= 2.7.0 (confirmed: 2.7.8.0)
- Use `--attention-backend flashinfer` (FlashAttention 3 not stable on
  Blackwell WSL2 yet)
- Use `--dtype float16` with `--quantization awq_marlin`
- Do NOT use `--enforce-eager` (disables CUDA graphs, 8x slower)
- FP8 is NOT supported via WSL2 dxgkrnl yet — stay on AWQ-Marlin

## Coding Standards (Non-Negotiable)

These are hard rules. Never deviate from them.

1. **Single quotes** in all Python code — `'value'` not `"value"`
2. **No argparse** anywhere — use `input()` with `while True` for any CLI interaction
3. **No Unicode or em-dashes** in any source file — ASCII only in code and comments
4. **Every file starts with its path as a comment on line 1**
   ```python
   # src/ingestion/parser.py
   ```
5. **Complete files only** — never write partial files, never use `# ... rest unchanged`,
   never truncate with `# etc`. Every file delivered must be 100% complete and runnable.
6. **No markdown in source files** — no bold, no headers inside .py files
7. **Clean prose** for any user-facing strings — no markdown symbols in UI text

## Technology Stack

### Models (all local, pre-downloaded to /models/)
| Role | Model | Path |
|---|---|---|
| LLM | Qwen3-14B-AWQ | /models/Qwen3-14B-AWQ |
| Embeddings | BGE-M3 | /models/bge-m3 |
| Reranker | BGE-Reranker-v2-m3 | /models/bge-reranker-v2-m3 |
| NLI guard | cross-encoder/nli-deberta-v3-small | /models/nli-deberta-v3-small |

### Infrastructure
| Component | Choice |
|---|---|
| LLM runtime | vLLM 0.9+ (OpenAI-compatible, localhost:11436) |
| Vector DB | Qdrant (in-process, in-memory, session-ephemeral) |
| Sparse index | rank-bm25 |
| PDF parser | Docling (primary) + PyMuPDF (fallback) |
| OCR | RapidOCR (ONNX + CUDA execution provider) |
| Backend | FastAPI + Uvicorn (localhost:8000) |
| Frontend | React + Vite + TailwindCSS (localhost:5173) |
| Orchestration | Docker Compose v2 |

### Key Python Libraries
```
docling>=2.0
pymupdf>=1.24
rapidocr-onnxruntime
qdrant-client>=1.9
rank-bm25
sentence-transformers>=3.0
torch>=2.9 (cu128 build)
transformers>=4.45
fastapi>=0.115
uvicorn[standard]
langchain>=0.3
openai  # for vLLM OpenAI-compat client
pydantic>=2.0
python-multipart
aiofiles
cryptography  # AES-256 session encryption
pytest
deepeval
```

## Project Structure

```
board-intelligence/
├── CLAUDE.md                    <- you are here
├── ARCHITECTURE.md              <- full EDD reference
├── docker-compose.yml           <- orchestration (exists)
├── Dockerfile.vllm              <- vLLM CUDA 12.8 container (exists)
├── Dockerfile.backend           <- FastAPI + models container
├── Dockerfile.frontend          <- React/Vite container
├── .env.example                 <- environment variable template
├── requirements.txt             <- pinned Python deps
├── models/                      <- model weights (downloaded separately)
├── scripts/
│   ├── download_models.py       <- downloads all 4 models to /models/
│   └── verify_gpu.py            <- confirms CUDA 12.8 + sm_120 setup
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py            <- Docling + PyMuPDF fallback
│   │   ├── ocr.py               <- RapidOCR ONNX/CUDA
│   │   ├── table_extractor.py   <- TableFormer table handling
│   │   └── chunker.py           <- semantic + sliding overlap chunker
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embedder.py          <- BGE-M3 batched CUDA embedding
│   │   ├── bm25_index.py        <- rank-bm25 sparse index
│   │   └── vector_store.py      <- Qdrant in-memory store
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py  <- dense + sparse parallel async
│   │   ├── rrf_fusion.py        <- Reciprocal Rank Fusion (k=60)
│   │   └── reranker.py          <- BGE-Reranker-v2-m3 cross-encoder
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py    <- structured CoT prompt assembly
│   │   ├── llm_client.py        <- vLLM OpenAI-compat async client
│   │   ├── hallucination_guard.py <- NLI DeBERTa verifier
│   │   └── citation_injector.py <- [Page N] citation validation
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              <- FastAPI app, localhost:8000 only
│   │   ├── session.py           <- ephemeral AES-256 session store
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── ingest.py        <- POST /ingest (PDF upload)
│   │       ├── query.py         <- POST /query (Q&A with streaming)
│   │       └── health.py        <- GET /health
│   └── utils/
│       ├── __init__.py
│       ├── config.py            <- Pydantic Settings
│       ├── logger.py            <- structured logging
│       └── security.py          <- PDF hashing, AES utils
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_rrf_fusion.py
│   │   └── test_citation_injector.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── rag_eval/
│       ├── golden_set.json
│       └── eval_runner.py
└── frontend/
    ├── Dockerfile.frontend
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── components/
            ├── ChatPane.jsx
            ├── SourcePanel.jsx
            ├── UploadDropzone.jsx
            └── StatusBar.jsx
```

## Implementation Phases

### Phase 1 — Core Pipeline (implement first)
Goal: PDF in -> question in -> cited answer out, end-to-end working.

Order of implementation:
1. `scripts/verify_gpu.py` -- confirm environment before anything else
2. `scripts/download_models.py` -- pull all 4 models to /models/
3. `src/utils/config.py` -- Pydantic settings, read from env
4. `src/utils/logger.py` -- structured logging
5. `src/utils/security.py` -- PDF SHA-256 hash, AES-256-GCM utils
6. `src/ingestion/parser.py` -- Docling primary, PyMuPDF fallback
7. `src/ingestion/ocr.py` -- RapidOCR ONNX/CUDA
8. `src/ingestion/table_extractor.py` -- table handling
9. `src/ingestion/chunker.py` -- semantic + sliding overlap
10. `src/indexing/embedder.py` -- BGE-M3 batched CUDA
11. `src/indexing/bm25_index.py` -- rank-bm25
12. `src/indexing/vector_store.py` -- Qdrant in-memory
13. `src/retrieval/hybrid_retriever.py` -- dense + sparse async
14. `src/retrieval/rrf_fusion.py` -- RRF k=60
15. `src/retrieval/reranker.py` -- BGE-Reranker-v2-m3
16. `src/generation/prompt_builder.py` -- structured prompt
17. `src/generation/llm_client.py` -- vLLM async client
18. `src/generation/hallucination_guard.py` -- NLI verifier
19. `src/generation/citation_injector.py` -- citation validation
20. `src/api/main.py` + routes + session
21. `Dockerfile.backend` + `Dockerfile.frontend`
22. `requirements.txt`
23. Frontend React components
24. `tests/` -- full test suite

### Phase 2 — Retrieval Quality
- BM25 tuning
- HyDE query expansion
- RAG eval with deepeval golden set

### Phase 3 — Production Hardening
- Numerical hallucination checker
- Per-page quality scoring
- Error boundary UI
- Air-gap bundle script

## Pipeline Data Flow

```
PDF file
  |
  v
[parser.py] -- render-mode detection (native vs scanned)
  |               |
  |          [ocr.py] -- RapidOCR CUDA on scanned pages
  |               |
  +---------------+
  |
  v
[table_extractor.py] -- TableFormer, tables -> structured Markdown
  |
  v
[chunker.py] -- semantic boundaries + 10% overlap, metadata enrichment
  |             chunk schema: {text, page_number, chunk_index,
  |                            section_heading, chunk_type, doc_hash}
  v
[embedder.py] -- BGE-M3 FP16 CUDA, batch_size=64
  |
  +-----------> [bm25_index.py] -- sparse index built in parallel
  |
  v
[vector_store.py] -- Qdrant in-memory, HNSW, cosine similarity
  |
  v  (at query time)
[hybrid_retriever.py] -- dense top-50 + sparse top-50 (async parallel)
  |
  v
[rrf_fusion.py] -- Reciprocal Rank Fusion k=60 -> merged top-50
  |
  v
[reranker.py] -- BGE-Reranker-v2-m3 cross-encoder -> top-5
  |
  v
[prompt_builder.py] -- system prompt + chunks + metadata + CoT instruction
  |
  v
[llm_client.py] -- vLLM streaming, Qwen3-14B-AWQ, max_tokens=1024
  |
  v
[citation_injector.py] -- validate [Page N] tags, flag orphan citations
  |
  v
[hallucination_guard.py] -- NLI DeBERTa async, flag unverified sentences
  |
  v
FastAPI /query endpoint -- SSE streaming response to frontend
```

## Security Rules (enforce in every file)

- FastAPI MUST bind to `127.0.0.1` only -- never `0.0.0.0` in production mode
- No model or embedding call may use an external URL or API key
- PDF bytes must be hashed (SHA-256) immediately on upload, stored in session
- Temp files go to `/tmp/bis-sessions/{session_id}/` with permissions 0o600
- Session cleanup: `shutil.rmtree` on session end or timeout (30 min)
- No logging of PDF content or user queries to disk in production mode
- Qdrant collection is in-memory only -- never persisted across sessions

## RAG Quality Targets

These are the acceptance criteria for Phase 1:

| Metric | Target |
|---|---|
| Ingest time (100-page PDF) | < 120 seconds |
| Query to first token | < 4 seconds |
| Full response (200 words) | < 12 seconds |
| Faithfulness (deepeval) | > 0.80 |
| Citation coverage | 100% factual sentences cited |
| Hallucination flag rate | > 95% on adversarial set |

## VRAM Budget Reference

| Component | VRAM |
|---|---|
| Qwen3-14B AWQ-Marlin | ~9.5 GB |
| BGE-M3 FP16 | ~1.1 GB |
| BGE-Reranker-v2-m3 FP16 | ~0.6 GB |
| NLI DeBERTa small | CPU (not VRAM) |
| KV cache (8K ctx) | ~4-5 GB |
| Working buffers | ~1-2 GB |
| **Total peak** | **~16-18 GB** |
| **Available** | **24 GB** |
| **Headroom** | **~6-8 GB** |

## Common Pitfalls to Avoid

1. Never import from `__future__` annotations in files that use Pydantic v2
2. Never use `model.encode()` without `device='cuda'` and `batch_size=64`
3. Never call Qdrant `.search()` without specifying `with_payload=True`
4. Never stream vLLM response without handling `finish_reason='stop'` explicitly
5. Never use `asyncio.run()` inside FastAPI route handlers -- use `await`
6. Docling: always call `DocumentConverter()` with `artifacts_path` set to
   `/models/docling-models` so it uses local weights, not downloaded ones
7. RapidOCR: initialize with `det_use_cuda=True, rec_use_cuda=True` explicitly
8. BGE-M3: use `BGEM3FlagModel` from `FlagEmbedding`, not sentence-transformers,
   for the full dense+sparse+colbert output
9. rank-bm25: tokenize with `.lower().split()` consistently at index and query time
10. Qdrant in-memory: use `QdrantClient(':memory:')` -- do NOT use `location=':memory:'`
    with the new client API (changed in qdrant-client 1.9+)
