# Board Intelligence

[![CI](https://github.com/0DevDutt0/board-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/0DevDutt0/board-intelligence/actions/workflows/ci.yml)

![Board Intelligence: Local AI Governance](examples/Board%20Intelligence_%20Local%20AI%20Governance.png)

Local AI for board PDFs, with page citations and checks for who reviewed, recommended, or approved a decision.

`The committee reviewed the project.` and `The board approved the project.` describe different decisions. Board Intelligence retrieves the evidence for each claim, checks it against the cited page, and flags unsupported changes in authority. Ask a question the source cannot answer and the assistant can abstain.

## Demo

<video src="Demo/demo.mp4" controls muted width="100%"></video>

If the player does not load, open the [24-second demo video](Demo/demo.mp4) directly. It shows a cited answer and an abstention from the running app. See [Privacy and verification](#privacy-and-verification) for the scope of local inference.

Start with the fictional [Harbor Works board minutes](examples/board-minutes.pdf). Ask **“What did the committee review, and what did the board approve?”** and compare the cited passages. Then ask **“What did the board decide about opening an overseas office?”** to see what happens when the minutes contain no decision.

The demo uses an edited excerpt of Sascha Ende’s *Happy Beats & Business Moves Vol. 10* under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Track source](https://ende.app/en/song/12875-happy-beats-business-moves-vol-10)

## How it works

FastAPI parses the PDF and builds session-scoped dense (BGE-M3 and in-memory Qdrant) and sparse (BM25) indexes. Reciprocal rank fusion and a cross-encoder reranker select evidence for Qwen3-14B on local vLLM. The React app streams the answer and displays matching source passages. A CPU-based natural language inference model and governance checks assess support after generation.

<details>
<summary>Architecture diagram</summary>

![Board Intelligence system architecture](System%20Architecture%20Diagram.png)

</details>

## Run on the reference hardware

The reference setup uses Windows 11, WSL2 with Ubuntu, Python 3.12, Node.js 20+, and an RTX 5090 Laptop GPU with 24 GB of VRAM. The GPU preflight checks for sm_120, a CUDA 12.8 PyTorch build, and at least 20 GB of VRAM. Allow about 50 GB of free disk. The preflight describes this configuration, and other GPU configurations need their own validation.

Run these commands from the repository root in PowerShell.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe scripts/download_models.py
Copy-Item .env.example .env
.\venv\Scripts\python.exe scripts/verify_gpu.py
.\scripts\start_all.ps1
```

Open <http://localhost:5173>. The first setup downloads models and may install vLLM, the WSL CUDA toolkit, and frontend dependencies. On the reference laptop, limiting WSL memory to 12 GB with 16 GB swap helped leave sufficient host RAM for GPU allocations.

## Test and evaluate

The CPU suite uses the fictional PDFs and model doubles to exercise ingestion, retrieval, streaming, verification, and cleanup. It runs without downloading GPU models.

```text
python -m pip install -r requirements-ci.txt
python -m pytest tests/unit tests/integration -q
npm --prefix frontend ci
npm --prefix frontend run build
```

The [evaluation](tests/rag_eval/README.md) covers 12 questions across three fictional board packs and 12 labeled governance claims. It reports retrieved-page recall, evidence coverage against annotated facts, abstention, and response timing. The governance replay compares the same claims and passages with the rule enabled and disabled and exposes each false positive and missed warning.

In the [RTX 5090 reference run](tests/rag_eval/reference-rtx5090.json), all nine answerable questions received credit and all three unanswerable questions triggered abstention. The governance rules caught all four seeded authority errors with zero false positives in the replay. The live verification model still flagged one answer that met the benchmark’s annotated criteria.

```powershell
python -m tests.rag_eval.eval_runner --guard-only    # CPU, no servers or model weights
python -m tests.rag_eval.eval_runner                 # start the local stack first
```

## Privacy and verification

The application is designed for local inference after setup. The API and vLLM bind to loopback by default. The model client rejects remote origins, ignores proxy environment variables, and uses local model assets. The browser uses system fonts. Scanned PDFs need pre-staged Docling and OCR artifacts.

The browser sends a PDF body that the API processes in memory. Legacy multipart clients can trigger temporary spooling by the HTTP framework. Session metadata and retained chat history are AES-256-GCM encrypted in process memory. The active indexes and passages stay readable in process memory for retrieval and verification. End Session, inactivity expiry, and server shutdown release session state. Behavior under OS swap, browser storage, and process dumps depends on the host configuration.

Streamed text is provisional. After it arrives, the guard checks each claim against every page it cites and reports **Supported by cited evidence**, **Needs review**, **Insufficient evidence**, or **Verification unavailable**. The checks are model-based and heuristic. Review the cited passage before acting on an answer.

## Usage

Portfolio source is available for review. The project retains its internal-use restriction.
