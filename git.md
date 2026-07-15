# Repository Profile

Ready-to-paste details for publishing this project on GitHub.

---

## Repo Name

```
board-intelligence
```

Alternative names if taken: `board-intelligence-rag`, `offline-board-rag`, `boardroom-ai`

## Short Description (GitHub "About" field, max ~350 chars)

```
Fully offline, GPU-accelerated RAG system for confidential board documents. Hybrid retrieval (BM25 + BGE-M3 dense + RRF + cross-encoder reranking), Qwen3-14B on vLLM, NLI hallucination guard with governance-aware verb checks, page-level citations, AES-256 ephemeral sessions. Zero network egress.
```

## One-Line Tagline

```
Ask questions about a confidential board PDF. Get cited, verified answers. Nothing ever leaves the machine.
```

## Topics / Tags

```
rag  llm  vllm  retrieval-augmented-generation  hybrid-search  bm25  vector-search
qdrant  fastapi  react  offline-first  hallucination-detection  nli  cuda
document-ai  pdf-parsing  enterprise-ai  local-llm  qwen  privacy
```

## Skills Demonstrated

### AI / ML Engineering
- End-to-end RAG pipeline design: parsing -> chunking -> hybrid retrieval -> reranking -> generation -> verification
- Hybrid retrieval: BM25 sparse + BGE-M3 dense (1024-dim) fused with Reciprocal Rank Fusion (k=60)
- Cross-encoder reranking (BGE-Reranker-v2-m3) with score-threshold filtering
- HyDE query expansion with domain-anchored prompting and think-block stripping
- Semantic chunking via embedding cosine-similarity topic-shift detection with sliding overlap
- NLI-based hallucination detection (DeBERTa-v3) with sentence-level premise decomposition
- Novel governance-aware verification: verb-inflation detection (advisory "reviews" vs executive
  "approves"), event attribution checks, citation-overlap validation
- Local LLM serving: vLLM, AWQ-Marlin quantization, CUDA graphs, FlashInfer attention
- RAG evaluation: 15-item golden set, keyword/page recall, citation coverage, abstention accuracy

### Systems / Infrastructure
- CUDA on Blackwell (sm_120): CUDA 12.8 toolchain pinning, flashinfer JIT patching,
  user-local toolkit install without root (dpkg -x)
- WSL2 GPU internals: dxgkrnl memory-commit behavior, WDDM backing-store diagnosis,
  .wslconfig memory rebalancing to fix phantom CUDA OOMs
- Windows-native ML serving: DLL load-order debugging (pyarrow access violation),
  cross-OS orchestration (PowerShell + WSL bash launchers with health-check sequencing)
- VRAM budget engineering: 4 models co-resident in 24 GB (LLM + embedder + reranker + KV cache)

### Backend / Full Stack
- FastAPI with SSE streaming (token stream -> citations -> corrected text -> verification events)
- Async pipeline: parallel dense/sparse retrieval, non-blocking post-generation verification
- Security: AES-256-GCM encrypted ephemeral sessions, PBKDF2 key derivation, SHA-256 document
  hashing, zero-fill file deletion, localhost-only binding, 30-min session expiry
- React 18 + Vite + Tailwind frontend: streaming chat, source panel with match-quality labels,
  error boundaries, drag-and-drop upload

### Engineering Practice
- 23 automated tests (unit + integration + multi-turn), fully local eval harness
- Pydantic Settings config, structured logging, graceful multi-level fallbacks
  (Docling -> PyMuPDF; reranker -> RRF order; retrieval -> full-context)

## Key Numbers (for resume bullets)

| Metric | Value |
|---|---|
| Backend Python (src/) | ~2,400 LOC across 31 files |
| Tests | 23 (1,100+ LOC), plus 15-item RAG eval golden set |
| Frontend | ~1,500 LOC React/JSX |
| Ops scripts | ~700 LOC (PowerShell + bash + Python) |
| Query -> first token | ~2.4 s |
| 3-page PDF ingest | ~4 s |
| Keyword / page recall | 100% on golden set |
| Citation coverage | 98.8% of factual bullets |
| Abstention accuracy | 100% on adversarial questions |
| LLM throughput | 140-160 tok/s (Qwen3-14B-AWQ, RTX 5090) |

## Suggested Resume Bullet

```
Built a fully offline RAG system for confidential board documents (Python/FastAPI/React,
vLLM + Qwen3-14B on RTX 5090): hybrid BM25 + dense retrieval with RRF fusion and
cross-encoder reranking, streaming SSE answers with page-level citations, and a
governance-aware NLI hallucination guard -- 100% page recall and 98.8% citation
coverage on a domain golden set, with zero network egress.
```
