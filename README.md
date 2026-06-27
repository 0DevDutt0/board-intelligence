# Board Intelligence System

An enterprise-grade, fully offline Retrieval-Augmented Generation (RAG) system for board meeting intelligence. Upload a confidential board PDF at session start and get cited, natural-language answers — entirely on local hardware with zero network egress and zero external API calls.

---

## Overview

Board meetings generate dense, high-stakes documents: financial summaries, risk disclosures, strategic resolutions, governance decisions. Extracting specific facts quickly is hard. Board Intelligence solves this by ingesting the PDF into a hybrid retrieval pipeline, then answering questions with page-level citations verified against the source text.

Every component runs locally. No data leaves the machine.

---

## How It Works

The system implements a five-layer RAG pipeline that transforms a raw PDF into streamed, cited answers.

```
PDF Upload
    |
    v
[1] Document Ingestion
    - Render-mode detection (native text vs. scanned image)
    - Docling (DocLayNet + TableFormer) for layout-aware parsing
    - RapidOCR (ONNX/CUDA) for scanned pages
    - Semantic chunking with 10% sliding overlap
    |
    v
[2] Embedding + Indexing
    - BGE-M3 dense embeddings (1024-dim, FP16, CUDA, batch=64)
    - Qdrant in-memory HNSW index (cosine similarity)
    - BM25 sparse index built in parallel
    |
    v
[3] Hybrid Retrieval + Reranking  (at query time)
    - Dense: Qdrant HNSW top-50
    - Sparse: BM25 keyword top-50
    - Fusion: Reciprocal Rank Fusion (k=60) -> merged top-50
    - Reranker: BGE-Reranker-v2-m3 cross-encoder -> top-12, then
      filtered by score threshold (>= 0.10, min 3 kept)
    - BM25 acronym expansion: GDA/GIC/MCS/HSSC and 15+ governance
      acronyms expanded bidirectionally before tokenization so both
      abbreviated and full-form chunks are matched
    - Optional HyDE: queries >= 6 words generate a governance-domain
      hypothetical answer, concatenated with the original query before
      dense embedding for improved recall without losing keyword anchoring
    |
    v
[4] Augmented Generation
    - Structured prompt: system rules + up to 12 retrieved passages + question
    - Chain-of-thought instruction for multi-part questions
    - Qwen3-14B-AWQ via vLLM (AWQ-Marlin, FP16, FlashInfer attention)
    - Streamed token output over SSE
    |
    v
[5] Post-Generation Verification
    - CITATION CONSTRAINT injected into system prompt listing only the
      exact page numbers from retrieved chunks; LLM may not cite others
    - Citation validation: regex-extract [Page N] tags, replace any
      not in the retrieved set with [unverifiable citation]
    - Corrected text sent to frontend via SSE 'corrected' event,
      replacing the raw streamed output with the validated version
    - NLI hallucination guard (DeBERTa-v3-small, CPU, async):
      per-sentence entailment against retrieved passages (threshold 0.70)
    - Governance verb check: flags authority inflation ("approves" when
      source says "advises") and scope broadening ("manages" vs "reviews");
      skips sentences that already name the Board/management as actor
    - Numeric check: digit-form numbers (>= 4 digits) and word-form
      quantities ("four", "third", "twenty") must appear in source chunks
    - Structured warnings: each warning carries {message, snippet} so
      the UI can link it to the specific bullet that triggered it;
      messages written as correction prose ("approves" -- the source
      says the committee advises the Board here...)
    |
    v
FastAPI SSE stream -> React frontend
    - Amber left border on bullets whose verb is flagged
    - Source panel shows "Strong / Good / Weak match" quality labels
    - React ErrorBoundary catches JS crashes with retry UI
    - Error SSE events render as styled alert with recovery hint
```

---

## Retrieval Algorithm: Reciprocal Rank Fusion

Dense and sparse retrieval return independent ranked lists. RRF combines them without requiring score normalization:

```
rrf_score(chunk) = sum over each list of: 1 / (k + rank_in_that_list)
```

With `k=60`, a chunk ranked first in both lists scores `2/61 ≈ 0.033`. Chunks appearing in only one list get `1/61`. The merged list is re-ranked by descending fused score, then the top-50 go to the cross-encoder reranker which produces a final top-12 shortlist, further filtered by a minimum score threshold (0.10) before generation.

This approach consistently outperforms single-modality retrieval: BM25 handles exact terminology and acronyms; dense retrieval handles semantic paraphrase; the cross-encoder reranker does precision scoring on the shortlist.

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| LLM | Qwen3-14B-AWQ | AWQ-Marlin quantization, ~9.5 GB VRAM |
| LLM Runtime | vLLM 0.9+ | OpenAI-compatible API, localhost:11436 |
| Embeddings | BGE-M3 (FlagEmbedding) | Dense + sparse + ColBERT, 1024-dim, 1.1 GB VRAM |
| Reranker | BGE-Reranker-v2-m3 | Cross-encoder, 0.6 GB VRAM |
| Hallucination Guard | DeBERTa-v3-small (NLI) | CPU inference, async, does not block streaming |
| PDF Parser | Docling + PyMuPDF | Docling primary; PyMuPDF fallback |
| OCR | RapidOCR (ONNX/CUDA) | For scanned pages |
| Vector Store | Qdrant (in-memory) | HNSW, cosine, session-ephemeral |
| Sparse Index | rank-bm25 | BM25Okapi, k1=1.2 b=0.5, `.lower().split()` tokenization |
| Backend | FastAPI + Uvicorn | localhost:8000, SSE streaming |
| Frontend | React 18 + Vite + Tailwind | localhost:5173 |
| Orchestration | Docker Compose v2 | Three-service stack |
| Session Security | AES-256-GCM | In-memory encrypted state, 30-min timeout |

---

## Hardware Requirements

| Resource | Minimum | This System |
|---|---|---|
| GPU | Any CUDA 12.x-capable GPU with 16+ GB VRAM | RTX 5090 Laptop (24 GB) |
| CPU RAM | 16 GB | 23 GB |
| Disk | 50 GB free (models + data) | ~898 GB |
| OS | Linux or WSL2 | Windows 11 + WSL2 2.7.8 |
| CUDA Build | 12.8 | 12.8 (mandatory for sm_120) |

> **RTX 5090 / Blackwell (sm_120) note:** CUDA 13.x causes MMQ kernel segfaults on Blackwell. The entire stack is built against CUDA 12.8 explicitly. The NVIDIA driver supporting 13.x does not mean you should compile against it.

### VRAM Budget

| Component | VRAM |
|---|---|
| Qwen3-14B AWQ-Marlin | ~9.5 GB |
| BGE-M3 FP16 | ~1.1 GB |
| BGE-Reranker FP16 | ~0.6 GB |
| KV cache (8K context) | ~4-5 GB |
| Working buffers | ~1-2 GB |
| **Total peak** | **~16-18 GB** |
| **Headroom on 24 GB** | **~6-8 GB** |

---

## Project Structure

```
board-intelligence/
├── CLAUDE.md                    # Coding standards and implementation spec
├── ARCHITECTURE.md              # Full engineering design document
├── docker-compose.yml           # Three-service orchestration
├── Dockerfile.vllm              # vLLM CUDA 12.8 container
├── Dockerfile.backend           # FastAPI + models container
├── Dockerfile.frontend          # React/Vite container
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Environment variable template
├── models/                      # Model weights (downloaded separately)
├── scripts/
│   ├── download_models.py       # Download all 4 models to /models/
│   ├── verify_gpu.py            # Confirm CUDA 12.8 + sm_120 setup
│   └── bundle_airgap.py         # Build offline deployment tarball
├── src/
│   ├── ingestion/
│   │   ├── parser.py            # Docling primary, PyMuPDF fallback
│   │   ├── ocr.py               # RapidOCR ONNX/CUDA
│   │   ├── table_extractor.py   # TableFormer table-to-Markdown
│   │   └── chunker.py           # Semantic + sliding overlap chunker
│   ├── indexing/
│   │   ├── embedder.py          # BGE-M3 batched CUDA embedding
│   │   ├── bm25_index.py        # rank-bm25 sparse index
│   │   └── vector_store.py      # Qdrant in-memory store
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Dense + sparse parallel async + HyDE
│   │   ├── rrf_fusion.py        # Reciprocal Rank Fusion (k=60)
│   │   └── reranker.py          # BGE-Reranker-v2-m3 cross-encoder
│   ├── generation/
│   │   ├── prompt_builder.py    # Structured CoT prompt assembly
│   │   ├── llm_client.py        # vLLM async streaming client
│   │   ├── hallucination_guard.py  # NLI DeBERTa verifier
│   │   └── citation_injector.py    # [Page N] tag validation
│   ├── api/
│   │   ├── main.py              # FastAPI app, startup model loading
│   │   ├── session.py           # Ephemeral AES-256-GCM session store
│   │   └── routes/
│   │       ├── ingest.py        # POST /ingest
│   │       ├── query.py         # POST /query (SSE streaming)
│   │       └── health.py        # GET /health
│   └── utils/
│       ├── config.py            # Pydantic Settings (env-configurable)
│       ├── logger.py            # Structured stdout logger
│       ├── security.py          # SHA-256 PDF hash, AES-256-GCM utils
│       └── acronym_expander.py  # BM25 governance acronym expansion
├── tests/
│   ├── unit/
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_rrf_fusion.py
│   │   └── test_citation_injector.py
│   ├── integration/
│   │   └── test_pipeline.py     # End-to-end pipeline test
│   └── rag_eval/
│       ├── golden_set.json      # 15 Holcim-specific Q&A pairs
│       └── eval_runner.py       # Fully local metrics runner (no external API)
└── frontend/
    ├── src/
    │   ├── App.jsx              # Layout and session state
    │   ├── main.jsx
    │   └── components/
    │       ├── ChatPane.jsx      # SSE streaming chat with citation links
    │       ├── SourcePanel.jsx   # Retrieved passage display
    │       ├── UploadDropzone.jsx # PDF drag-and-drop upload
    │       └── StatusBar.jsx     # Session and performance stats
    ├── tailwind.config.js
    └── vite.config.js
```

---

## Setup

### 1. Prerequisites

- Docker 29+ with NVIDIA Container Toolkit
- WSL2 >= 2.7.0 (`wsl --update` from PowerShell if needed)
- NVIDIA driver >= 592.01
- ~50 GB free disk space for models

### 2. Download Models

Run this once before building Docker images. Requires internet access for the initial download only.

```bash
pip install huggingface_hub docling
python scripts/download_models.py
```

Downloads to `./models/`:
- `Qwen3-14B-AWQ` (~9.5 GB)
- `bge-m3` (~1.1 GB)
- `bge-reranker-v2-m3` (~0.6 GB)
- `nli-deberta-v3-small` (~180 MB)
- `docling-models` (DocLayNet + TableFormer)

### 3. Verify GPU Environment

```bash
python scripts/verify_gpu.py
```

All checks must pass before proceeding. Critical checks:
- CUDA available
- Compute capability sm_120 (RTX 5090)
- PyTorch built with CUDA 12.8
- VRAM >= 20 GB

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and set a strong SESSION_SECRET_KEY
```

### 5. Build and Run

```bash
docker compose up --build
```

Startup sequence:
1. `vllm` container starts and loads Qwen3-14B-AWQ (takes ~60-90 seconds)
2. `backend` container waits for vLLM health check, then loads BGE-M3, reranker, and NLI guard
3. `frontend` container starts Vite dev server

Access the UI at `http://localhost:5173`.

---

## Usage

### Web Interface

1. Open `http://localhost:5173`
2. Drag and drop a board meeting PDF (up to 100 MB)
3. Wait for ingestion to complete (shown in status bar)
4. Ask questions in natural language
5. Answers stream in real-time with `[Page N]` citation links
6. The right panel shows retrieved passages with "Strong / Good / Weak match" quality labels

### API

#### Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@board_minutes.pdf"
```

```json
{
  "session_id": "3f7a2c1e-...",
  "page_count": 42,
  "chunk_count": 187,
  "ingest_time_seconds": 38.4
}
```

#### Ask a Question (SSE streaming)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: 3f7a2c1e-..." \
  -d '{"question": "What was the capital expenditure approved?", "use_hyde": true}'
```

Response is a Server-Sent Events stream:

```
event: chunk
data: {"text": "The board approved"}

event: chunk
data: {"text": " a capital expenditure of $5.2 million [Page 3]"}

event: citations
data: {"chunks": [{"page_number": 3, "text": "...", "reranker_score": 4.21}, ...]}

event: corrected
data: {"text": "...full validated answer with [unverifiable citation] replacements..."}

event: verification
data: {"warnings": [{"message": "Authority transfer: source says advises...", "snippet": "Approves sustainability strategy"}]}

event: done
data: {"total_tokens": 84, "latency_ms": 3210}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "vllm_ready": true,
  "models_loaded": true,
  "gpu_memory_used_mb": 14823
}
```

#### End a Session

```bash
curl -X DELETE http://localhost:8000/session/3f7a2c1e-...
```

Session cleanup zero-fills the PDF temp file and removes all in-memory state.

---

## Chunking Strategy

The chunker uses semantic boundary detection rather than fixed-size windows:

1. Split page text into sentences
2. Embed all sentences with BGE-M3 (batch, lightweight 128-token pass)
3. Compute cosine similarity between adjacent sentences
4. Mark a chunk boundary where similarity drops below 0.70 (topic shift)
5. Apply 50-token sliding overlap between chunks to preserve context across boundaries
6. Tables are treated as atomic units and never split mid-row; only row-group splits are permitted when a table exceeds 1024 tokens

Each chunk carries full metadata:

```python
{
    "text": "...",
    "page_number": 3,
    "chunk_index": 12,
    "section_heading": "Capital Expenditure",
    "chunk_type": "text",       # or "table"
    "doc_hash": "sha256hex...",
    "token_count": 487,
    "embedding_prefix": "[Page 3, Section: Capital Expenditure]"
}
```

The `embedding_prefix` is prepended before encoding to give the model positional context without polluting the chunk text used for BM25 tokenization.

---

## Session Security

Sessions are ephemeral by design:

- Each uploaded PDF gets a `uuid4` session ID
- PDF bytes written to `/tmp/bis-sessions/{session_id}/document.pdf` with `chmod 0600`
- Session state is AES-256-GCM encrypted in memory (PBKDF2-derived key from session ID + server secret)
- Qdrant collection lives in memory only, never persisted
- Sessions expire after 30 minutes of inactivity
- On expiry or explicit DELETE: PDF temp file is zero-filled before `shutil.rmtree`
- The FastAPI server binds to `127.0.0.1` only; port 8000 is never exposed to the network

---

## Performance Targets

| Metric | Target | Latest Result |
|---|---|---|
| Ingest time (100-page PDF) | < 120 seconds | Passes |
| Query to first token | < 4 seconds | Passes |
| Full response (200 words) | < 12 seconds avg | 11.51s avg |
| Citation coverage | > 90% of bullets cited | 98.8% |
| Keyword recall | > 80% | 100.0% |
| Page recall | > 70% | 100.0% |
| Abstention accuracy | 100% on adversarial | 100.0% |
| Avg warnings (non-adversarial) | < 2.0 per query | 0.75 |

The RAG eval golden set (15 Holcim-specific Q&A pairs) is run fully locally with no external API calls. All targets are met as of the current build. Results reflect the Phase 2 + Phase 3 improvements including HyDE fix, acronym expansion, and guard calibration.

Typical performance on RTX 5090 with Qwen3-14B-AWQ + FlashInfer:
- LLM throughput: ~140-160 tokens/second
- Dense embedding (100-chunk batch): ~2-3 seconds
- Reranker (50 pairs, top-12): ~0.5 seconds
- NLI guard (per-sentence, async): ~1-2 seconds (does not block streaming)

---

## Running Tests

### Unit Tests (no GPU required)

```bash
# From the project root, with dependencies installed
pip install -r requirements.txt
pytest tests/unit/ -v
```

Unit tests mock the embedding model and GPU calls. They cover:
- RRF score correctness and edge cases
- Citation tag extraction and invalid page replacement
- Parser render-mode detection and fallback behavior
- Chunker schema, overlap, and table chunk type

### Integration Tests (requires live stack)

Place a test PDF at `tests/fixtures/sample_board_doc.pdf`, then:

```bash
docker compose up -d
BIS_API_URL=http://127.0.0.1:8000 pytest tests/integration/ -v
```

Integration tests verify:
- Ingest completes under 120 seconds
- Query returns SSE events including citations and done
- Unanswerable questions trigger the abstention response

### RAG Evaluation

```bash
python tests/rag_eval/eval_runner.py path/to/board.pdf
```

Runs the 15-item Holcim-specific golden set through the live stack. The evaluator is fully local — no deepeval, no OpenAI calls. It reports:

| Metric | How measured |
|---|---|
| Keyword recall | Golden keywords present in answer |
| Citation coverage | Fraction of bullets with a [Page N] tag |
| Page recall | Cited pages overlapping expected pages |
| Abstention accuracy | Adversarial questions correctly refused |
| Avg latency | Average wall-clock time per query |
| Warning rate | Avg hallucination guard warnings per non-adversarial question |

Results are written to `tests/rag_eval/results/eval_YYYYMMDD_HHMMSS.json`.

---

## Configuration Reference

All settings are read from environment variables or `.env`. Key tunables:

| Variable | Default | Description |
|---|---|---|
| `SESSION_SECRET_KEY` | (required) | AES key derivation secret; set a strong random value |
| `VLLM_BASE_URL` | `http://vllm:11436` | vLLM service URL |
| `VLLM_MAX_TOKENS` | `1024` | Max generation tokens per response |
| `DENSE_TOP_K` | `50` | Qdrant results before RRF |
| `SPARSE_TOP_K` | `50` | BM25 results before RRF |
| `RRF_K` | `60` | RRF k constant |
| `RERANKER_TOP_K` | `12` | Passages scored by reranker before score filtering |
| `RERANKER_MIN_SCORE` | `0.10` | Drop chunks below this score (min 3 always kept) |
| `BM25_K1` | `1.2` | BM25 term frequency saturation (higher = less saturation) |
| `BM25_B` | `0.5` | BM25 length normalization (lower = less penalty for long chunks) |
| `USE_HYDE` | `true` | Enable HyDE query expansion |
| `HYDE_MIN_QUERY_WORDS` | `6` | Minimum query words to trigger HyDE |
| `HYDE_CONCAT_QUERY` | `true` | Prepend original query to hypothetical before embedding |
| `NLI_ENTAILMENT_THRESHOLD` | `0.70` | Sentence verification threshold |
| `CHUNK_TARGET_TOKENS` | `500` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between adjacent chunks |
| `SESSION_TIMEOUT_MINUTES` | `30` | Session inactivity timeout |
| `MAX_UPLOAD_MB` | `100` | Maximum PDF upload size |
| `LOG_QUERIES` | `false` | Set false in production (no query logging to disk) |

---

## Architecture Decisions

**Why Docling over PyMuPDF as primary?**
Docling uses DocLayNet for layout detection and TableFormer for structured table extraction. It correctly identifies section boundaries, column layouts, and table cells that PyMuPDF treats as raw text. PyMuPDF is retained as a fast fallback.

**Why BGE-M3 over OpenAI Ada or similar?**
BGE-M3 produces dense embeddings, sparse lexical weights, and ColBERT multi-vector representations from a single forward pass. It runs locally. Its 8192-token context window handles long passages without truncation.

**Why vLLM over llama.cpp?**
vLLM uses CUDA graphs and PagedAttention for efficient KV cache management, enabling higher throughput and lower latency on large batches. On Blackwell (sm_120) with FlashInfer attention, it achieves 140-160 tok/s on Qwen3-14B-AWQ vs roughly 40-60 tok/s with llama.cpp.

**Why AES-256-GCM for session state?**
Board documents are confidential. Encrypting in-memory session state ensures that even a memory dump during an active session does not expose the conversation history or document metadata in plaintext. The encryption key is derived per-session via PBKDF2.

**Why RRF over learned fusion?**
RRF requires no training data, has no hyperparameters that drift with document distribution, and is provably robust across query types. The constant `k=60` was established in the original paper and remains the standard default.

---

## Air-Gap Deployment

To deploy the system on a machine with no internet access:

```bash
# On the internet-connected build machine (after docker compose up --build):
python scripts/bundle_airgap.py
```

This script:
1. Verifies all model weights are present in `./models/`
2. Builds all three Docker images
3. Saves each image as a `.tar` file
4. Copies source files and configuration templates
5. Writes a `load_and_run.sh` helper script
6. Packages everything as `dist/board-intelligence-YYYYMMDD_HHMMSS.tar.gz`

On the air-gapped target machine:

```bash
tar -xzf board-intelligence-*.tar.gz
cd board-intelligence-bundle
./load_and_run.sh
# Edit .env to set SESSION_SECRET_KEY, then docker compose up
```

The bundle does not include model weights — transfer `./models/` separately if the target machine has never run the system before.

---

## License

Internal use only. Not for distribution.
