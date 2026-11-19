# ARCHITECTURE.md — Board Intelligence System
# Full Engineering Design Document
# Reference document for implementation decisions.

## 1. System Overview

Local RAG for board documents. Each session processes one PDF and answers
natural-language questions with citations. Model inference runs locally after
setup. See the [README](README.md#privacy-and-verification) for the runtime
privacy boundary.

Hardware: RTX 5090 Laptop (sm_120, 24 GB VRAM), 23 GB RAM, WSL2 2.7.8.0.

## 2. Architecture Layers

### Layer 1: Document Ingestion
- PDF render-mode detection (native text vs scanned)
- Docling: DocLayNet layout detection, TableFormer table extraction
- RapidOCR ONNX/CUDA for scanned pages
- Semantic chunker with 10% sliding overlap (400-600 tokens target)
- Metadata: page_number, chunk_index, section_heading, chunk_type, doc_hash

### Layer 2: Embedding and Indexing
- BGE-M3 (FlagEmbedding): dense + sparse + multi-vector from one model
- Batch size 64, FP16, CUDA
- Qdrant in-memory: HNSW index, cosine similarity
- rank-bm25: parallel sparse index
- Contextual prefix: "[Page N, Section: heading]" prepended before embedding

### Layer 3: Hybrid Retrieval and Reranking
- Dense: Qdrant HNSW ANN top-50
- Sparse: BM25 keyword top-50
- Fusion: Reciprocal Rank Fusion (RRF) k=60 -> merged top-50
- Cross-encoder: BGE-Reranker-v2-m3 -> top-5
- Optional HyDE: for short queries (<6 words), generate hypothetical answer
  first, embed that for improved recall

### Layer 4: Augmented Generation
- Prompt: system instructions + [Page N] citation rules + 5 chunks + query
- Chain-of-thought: "Think step by step" for multi-part questions
- LLM: Qwen3-14B-AWQ via vLLM, streamed, max_tokens=1024
- Post-gen: citation tag validation, NLI hallucination guard, numeric check

### Layer 5: Interface
- FastAPI backend, localhost:8000, SSE streaming for responses
- Ephemeral session (30 min timeout), AES-256-GCM encrypted state
- React frontend, localhost:5173
- Chat pane: streamed answer with inline [Page N] citation links
- Source panel: top-5 retrieved chunks, reranker scores, verbatim text

## 3. Model Specifications

### Qwen3-14B-AWQ
- Context: 32,768 tokens
- Quantization: AWQ-Marlin (not FP8 -- WSL2 dxgkrnl doesn't expose FP8)
- VRAM: ~9.5 GB
- Throughput: ~140-160 tok/s on RTX 5090 with CUDA graphs + FlashInfer
- vLLM flags: --quantization awq_marlin --dtype float16
  --attention-backend flashinfer --gpu-memory-utilization 0.60

### BGE-M3
- Architecture: XLM-RoBERTa base, 570M params
- Output: dense (1024-dim) + sparse (vocab-size) + colbert multi-vector
- Max tokens: 8192
- VRAM: ~1.1 GB FP16
- Library: FlagEmbedding (BGEM3FlagModel), NOT sentence-transformers
- batch_size=64 at ingest time

### BGE-Reranker-v2-m3
- Architecture: XLM-RoBERTa, cross-encoder
- Input: (query, document) pairs
- Output: relevance score [-inf, +inf], higher = more relevant
- VRAM: ~0.6 GB FP16
- Library: sentence-transformers CrossEncoder

### NLI DeBERTa small
- Model: cross-encoder/nli-deberta-v3-small
- Purpose: verify each response sentence against retrieved chunks
- Run on CPU (async, does not block response streaming)
- Labels: ENTAILMENT / NEUTRAL / CONTRADICTION
- Threshold: entailment probability > 0.70 to consider verified

## 4. Chunking Details

Strategy: semantic chunking using BGE-M3 embeddings to detect topic shifts.

Parameters:
- target_chunk_tokens: 500
- min_chunk_tokens: 100
- overlap_tokens: 50 (10% of 500)
- table_max_tokens: 1024 (tables are atomic, split only at row groups)

Chunk metadata schema:
```python
{
    'text': str,
    'page_number': int,
    'chunk_index': int,
    'section_heading': str,
    'chunk_type': str,  # 'text' or 'table'
    'doc_hash': str,    # SHA-256 of source PDF bytes
    'token_count': int,
    'embedding_prefix': str  # "[Page N, Section: heading]"
}
```

## 5. Retrieval Details

### Dense retrieval
- Query encoded with BGE-M3 dense head
- Qdrant HNSW search, top_k=50
- Distance: Cosine
- With payload: True (to recover metadata for RRF)

### Sparse retrieval
- Query tokenized with .lower().split()
- BM25 scores for all indexed chunks
- Top-50 by BM25 score

### RRF Fusion
```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

# For each chunk, sum RRF scores from dense and sparse rankings
# Re-rank by fused score, take top-50
```

### Reranking
- Input: query + top-50 fused chunks
- BGE-Reranker-v2-m3 scores all 50 as batch
- Sort by reranker score descending
- Take top-5 for generation

## 6. Prompt Template

System prompt (exact, do not modify without updating tests):

```
You are a precise board meeting intelligence assistant.
You answer questions strictly using the provided context passages
from the board document.

Rules:
- Cite every factual claim with [Page N] immediately after the claim.
- If the context does not contain enough information to answer,
  respond with exactly:
  "I cannot find sufficient information in the provided document
  to answer this question."
- Do not speculate or introduce information outside the context.
- For numerical data, reproduce figures exactly as they appear.
- Structure your answer clearly and concisely.

Context passages:
{context}

Question: {question}

Think step by step, then provide your final answer.
```

Context format per chunk:
```
--- Passage {i} [Page {page_number}, {chunk_type}] ---
{text}
```

## 7. Hallucination Guard Details

### NLI verification pass
For each answer claim, group the retrieved passages by the pages it cites.
Rank up to four candidate sentences per cited page and require at least one
candidate above the entailment threshold for each page. Numeric and governance
checks run on the cited evidence. A warning moves the answer to Needs review.

### Citation validation
1. Extract all [Page N] tags from response via regex
2. For each tag, verify N is a page in the retrieved evidence for the query
3. Invalid tags -> replace with [unverifiable citation]
4. Sentences with no citation tag -> flag for review

## 8. Session Management

The browser uploads the PDF body, which the API reads and parses in memory.
The application retains extracted chunks and indexes as live objects. The
compatibility multipart route may use a temporary spool managed by the HTTP
framework. The session ends after inactivity, an explicit DELETE, or shutdown.
Ending it releases both the encrypted state and the live indexes.

Session state (encrypted AES-256-GCM in memory):
```python
{
    'session_id': str,
    'doc_hash': str,
    'doc_filename': str,
    'page_count': int,
    'chunk_count': int,
    'conversation_history': list,  # last 3 exchanges only
    'created_at': datetime,
    'last_active': datetime,
    'ingest_stats': dict
}
```

## 9. CUDA Build Requirements

### MANDATORY for RTX 5090 (sm_120 Blackwell)

Base image: nvcr.io/nvidia/pytorch:25.02-py3 (CUDA 12.8)

Do NOT use:
- Any base image with CUDA 13.x
- FORCE_CUBLAS=ON in llama.cpp builds
- --enforce-eager in vLLM (disables CUDA graphs)
- FlashAttention 3 (not stable on Blackwell WSL2)
- FP8 quantization (WSL2 dxgkrnl doesn't expose FP8 hardware path)

vLLM environment variables:
- VLLM_FLASH_ATTN_VERSION=2
- PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

vLLM startup flags:
- --quantization awq_marlin
- --dtype float16
- --attention-backend flashinfer
- --gpu-memory-utilization 0.60
- --max-model-len 8192
- --swap-space 4

## 10. API Endpoints

### POST /ingest
- Body: multipart/form-data, field: 'file' (PDF)
- Max size: 100 MB
- Response: { session_id, page_count, chunk_count, ingest_time_seconds }
- Side effects: full ingestion pipeline runs synchronously (async task)

### POST /query
- Headers: X-Session-ID: {session_id}
- Body: { question: str, use_hyde: bool = true }
- Response: Server-Sent Events stream
  - event: chunk -> { text: str }
  - event: citations -> { chunks: list[CitationChunk] }
  - event: verification -> { warnings: list[str] }
  - event: done -> { total_tokens: int, latency_ms: int }

### GET /health
- Response: { status, vllm_ready, models_loaded, gpu_memory_used_mb }

### DELETE /session/{session_id}
- Explicit session cleanup
- Response: { deleted: bool }

## 11. Error Handling Strategy

Every pipeline stage must:
1. Catch and log exceptions with full stack trace
2. Never surface raw exceptions to the frontend
3. Degrade gracefully: parser fail -> fallback, retrieval fail -> empty result
4. Return structured error responses: { error: str, stage: str, recoverable: bool }

Specific cases:
- PDF parse failure: return partial results with page-level error flags
- vLLM timeout (>30s): return 503 with retry guidance
- CUDA OOM: log, attempt to free cache, retry once, then return 503
- Empty retrieval (no relevant chunks): LLM receives "no relevant context found"
  and is instructed to abstain

## 12. Testing Requirements

### Unit tests (pytest)
- test_parser.py: digital PDF, scanned PDF, mixed PDF, malformed PDF
- test_chunker.py: short doc, long doc, table-only doc, overlap correctness
- test_rrf_fusion.py: empty inputs, single-source, identical ranks
- test_citation_injector.py: valid tags, invalid page numbers, no citations

### Integration tests
- test_pipeline.py: end-to-end with synthetic 20-page test PDF
  - Ingest completes under 120 seconds
  - Query returns answer with citations under 12 seconds
  - Unanswerable question triggers abstention response

### RAG eval (deepeval)
Metrics measured on golden_set.json (50 Q&A pairs):
- AnswerCorrectness > 0.75
- Faithfulness > 0.80
- ContextRecall > 0.70
- CitationCoverage = 1.0 (all factual sentences cited)

Run: python tests/rag_eval/eval_runner.py
