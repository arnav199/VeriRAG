# VeriRAG

VeriRAG is a local-first research prototype for asking questions about PDFs and inspecting the evidence behind generated answers. It combines a conventional retrieval-augmented generation (RAG) pipeline with claim-level evidence assessments. It is an experiment in reliability; it does not eliminate hallucinations or guarantee that an answer is true.

## What it does

- Extracts text and page numbers from PDFs, splits text into chunks, and embeds those chunks locally.
- Stores vectors in a local FAISS index and retrieves relevant passages for a question.
- Sends retrieved passages to a configurable OpenAI-compatible chat completions API when an API key is configured.
- Extracts answer claims and assesses them with a cosine-similarity heuristic or an evidence-conditioned LLM verifier.
- Returns source filenames, page references, retrieved excerpts, labels, explanations, and latency.
- Keeps original documents and index files under `data/storage/` on this computer.

The application is designed for local, single-user research. An external LLM provider receives the question and retrieved passages. Do not use sensitive documents with an external provider unless its data handling is appropriate for your use.

## Architecture

```text
React/Vite browser
       │ HTTP JSON + PDF upload
FastAPI backend
       ├── PyMuPDF → page-aware chunks
       ├── Sentence Transformers → normalized embeddings
       ├── FAISS CPU → local similarity search
       ├── OpenAI-compatible LLM → answer, claims, optional verifier
       └── data/storage/ → PDFs, metadata, chunks, FAISS index
```

The similarity verifier is deliberately described as a heuristic. Cosine similarity says that two pieces of text are semantically close; it does not determine whether one proves the other. The LLM verifier can return `CONTRADICTED` separately from `UNSUPPORTED`, and can abstain with `INSUFFICIENT_EVIDENCE`.

## Prerequisites

- Windows with Python 3.13 and the `py` launcher. Python 3.12 is also suitable if the ML dependency combination on your machine requires it.
- Node.js and npm.
- Internet access for initial Python package installation and the first download of the embedding model.
- An API key is optional for installation and tests, but required for generated answers and LLM verification.

No Python packages are installed globally by these instructions. Direct Python dependencies are pinned in `requirements.txt` for repeatable installs; review and update them deliberately as the project evolves.

## Install

Run from the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, use `.\.venv\Scripts\python.exe` in place of `python` for the commands below. Add your API key to `.env` as `LLM_API_KEY=...`; do not commit `.env`. Or set `$env:LLM_API_KEY` directly in that PowerShell session.

Install the browser dependencies in a second terminal:

```powershell
cd frontend
npm install
```

The first use of Sentence Transformers downloads `sentence-transformers/all-MiniLM-L6-v2`. Embeddings run on CPU by default.

## Run locally

Start the API from the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

Open the local Vite URL shown in the terminal. The API is at `http://127.0.0.1:8000`; interactive API docs are at `/docs`. On first run, add one or more PDFs. Without `LLM_API_KEY`, uploads and retrieval remain available, while the ask endpoint returns a clear setup message.

### Configuration

Copy `.env.example` to `.env` in the repository root and set the values there. The backend loads that file at startup; shell environment variables take precedence.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_API_KEY` | External chat API key | unset |
| `LLM_BASE_URL` | OpenAI-compatible API base | `https://api.openai.com/v1` |
| `LLM_MODEL` | Provider model name | `gpt-4o-mini` |
| `VERIRAG_DATA_DIR` | Local documents and index directory | `./data/storage` |
| `VERIRAG_EMBEDDING_MODEL` | Sentence Transformers model ID | `sentence-transformers/all-MiniLM-L6-v2` |
| `VERIRAG_CHUNK_SIZE` / `VERIRAG_CHUNK_OVERLAP` | Character chunking controls | `900` / `120` |
| `VERIRAG_TOP_K` | Default number of retrieved chunks | `5` |
| `VERIRAG_MAX_UPLOAD_MB` | Upload size limit | `20` |
| `VERIRAG_VERIFIER` | Default `heuristic` or `llm` verifier | `heuristic` |

Documents are deduplicated by SHA-256 of file bytes. If changing the embedding model or its vector dimension, remove the local index and re-upload documents to rebuild it.

## API

- `GET /api/health` — backend status and whether an LLM key is configured.
- `GET /api/documents` — indexed document metadata.
- `POST /api/documents` — multipart PDF upload.
- `POST /api/retrieval` — inspect top-k passages for a query without generating an answer.
- `POST /api/ask` — generate an answer and claim assessments. Optional `top_k` and `verifier` fields are accepted.

The answer response keeps the original answer, cited chunks, retrieved evidence, claim assessments, verifier method, and elapsed time distinct.

## Tests

From the repository root, run:

```powershell
python -m pytest backend/tests -q
```

Tests use generated tiny PDFs and fake embedding/LLM interfaces; they do not make paid API calls or need credentials. The first dependency install may still download ML libraries, but tests do not download a sentence embedding model.

## Evaluation

1. Create the sample PDF from its readable source: `python backend/scripts/make_sample_pdf.py`.
2. Start VeriRAG and upload `data/sample_documents/sample_policy.pdf` in the browser.
3. From the repository root, run `python backend/scripts/run_evaluation.py --verifier heuristic`.
4. The script writes raw per-question JSONL to `data/evaluation/results.jsonl`. Run again with `--verifier llm` to compare methods on the same questions and corpus; LLM mode requires credentials.

The starter questions cover answerable, partially answerable, unanswerable, and conflicting evidence. They contain reference answers and text passages but no fabricated claim labels or retrieval chunk IDs. Human annotation should be added before reporting supervised scores.

### Metric definitions

- **Claim precision, recall, and F1:** one-vs-rest scores for a selected label, calculated only when expected claim labels are supplied and aligned with extracted claims. A zero denominator returns zero; verifier labels are not ground truth by themselves.
- **Unsupported-claim rate:** claims labeled `UNSUPPORTED` or `CONTRADICTED`, divided by all assessed claims. It depends on the corpus, annotation policy, retrieval, and verifier; it is not a universal hallucination rate.
- **Answer token F1:** token multiset overlap between a generated answer and a reference answer. This is a rough wording overlap, not semantic correctness.
- **Retrieval precision and recall:** retrieved chunk IDs intersecting manually labeled relevant chunk IDs, divided by retrieved IDs and relevant IDs respectively.
- **Citation support rate:** cited chunk IDs that match manually labeled relevant chunk IDs divided by all cited IDs.
- **Average latency:** mean API-reported elapsed milliseconds for successful answer requests.

Retrieval and citation metrics are omitted (`null`) until relevant chunk IDs are annotated. The JSONL rows preserve errors and raw outputs for pandas analysis. Compare baseline RAG and claim verification on the same questions; do not infer general performance from this small sample.

## Learning guide

See [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md) for beginner-friendly explanations of embeddings, cosine similarity, chunking, RAG, faithfulness, claim extraction, verification, and evaluation.

## Known limitations and next steps

- Scanned PDFs have no extractable text; OCR is not included.
- A local FAISS index and metadata files are appropriate for this prototype, not concurrent multi-user use or a production database.
- Prompt injection defenses instruct the model to treat excerpts as untrusted data, but prompt instructions alone cannot guarantee safe behavior.
- Similarity thresholds are not calibrated truth probabilities. LLM verification can be wrong or return invalid output.
- There is no claim that this implementation is novel or that verification improves reliability. Establish annotations and run controlled comparisons before making research claims.
- Next research step: refine the question “Does claim-level evidence verification improve the reliability of RAG-generated answers compared with a conventional RAG baseline?” and define the annotation policy before evaluating it.

