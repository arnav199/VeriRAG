# VeriRAG

> A deterministic demonstration mode and a conventional RAG baseline are available in the current prototype. Demo output is scripted and explicitly labelled; it is not LLM output.

VeriRAG is a local-first research prototype for asking questions about PDFs and inspecting the evidence behind generated answers. It combines a conventional retrieval-augmented generation (RAG) pipeline with claim-level evidence assessments. It is an experiment in reliability; it does not eliminate hallucinations or guarantee that an answer is true.

## What it does

- Extracts text and page numbers from PDFs, splits text into chunks, and embeds those chunks locally.
- Stores vectors in a local FAISS index and retrieves relevant passages for a question.
- Sends retrieved passages to a configurable OpenAI-compatible chat completions API when an API key is configured.
- Extracts answer claims, searches FAISS separately for each claim, and assesses each claim with either a cosine-similarity heuristic or an evidence-conditioned LLM verifier.
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
       ├── FAISS CPU → question retrieval for the answer
       ├── claim extraction → independent embedding and FAISS retrieval per claim
       ├── OpenAI-compatible LLM → answer, claims, optional evidence verifier
       └── data/storage/ → PDFs, metadata, chunks, FAISS index
```

Answer-level retrieval supplies context to the answer model and remains separate from claim-level retrieval. Once claims are extracted, each claim is embedded and sent to FAISS on its own. Each claim result returns `retrieved_evidence` with chunk ID, document ID, filename, page, text, and similarity; the existing `evidence` field identifies the passages the verifier selected. The answer-level `retrieved_evidence` and `citations` response fields remain available as before.

The LLM verifier receives only a claim and that claim's retrieved passages. It must return a label, evidence chunk IDs, and a non-empty explanation. The API rejects malformed output and IDs outside that claim's retrieved set; provider or parsing errors remain API errors and are never converted into verification labels. `INSUFFICIENT_EVIDENCE` is used when retrieval returns no passages or the passages do not let the verifier decide. `UNSUPPORTED` is the verifier's assessment that retrieved evidence does not support the claim; `CONTRADICTED` requires evidence assessed as conflicting with it.

The similarity verifier remains a comparison baseline. Its threshold labels indicate topical similarity only; they do not establish entailment and are not calibrated probabilities. Claim-specific retrieval can still miss relevant evidence, and the LLM can misclassify the evidence it receives.

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

After installation, the one-command Windows launcher starts both servers in separate terminals, checks readiness, and opens the application:

```powershell
.\start_verirag.ps1
```

Keep both server terminals open while using the app. Close them with `Ctrl+C` when finished. If a port is already occupied, the launcher reports the process and leaves it running. To start the services manually, use separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

```powershell
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173`. The API is at `http://localhost:8000`; interactive API docs are at `/docs`. Without `LLM_API_KEY`, the app selects its clearly labelled deterministic demo mode. The theme and chat history are saved in this browser.

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

`POST /api/ask` accepts `question`, optional `top_k`, `verifier` (`heuristic` or `llm`), `mode` (`live` or `demo`), and `pipeline` (`baseline` or `verirag`). Live mode is the default and reports a clear setup error when `LLM_API_KEY` is missing. Demo mode is deterministic and scripted for the bundled policy; the response includes `generation_mode: "deterministic_demo"` and a limitation that identifies demo output. Baseline mode returns answer-level retrieval and citations with an empty `claims` list. VeriRAG mode adds claim-specific candidate evidence and verifier-selected citations.

- `GET /api/health` — backend status and whether an LLM key is configured.
- `GET /api/documents` — indexed document metadata.
- `POST /api/documents` — multipart PDF upload.
- `POST /api/retrieval` — inspect top-k passages for a query without generating an answer.
- `POST /api/ask` — generate an answer and claim assessments. Optional `top_k` and `verifier` fields are accepted.

The answer response keeps the original answer, answer citations and retrieved passages, claim assessments, verifier method, and elapsed time distinct. Each claim includes its own `retrieved_evidence` candidate list and the verifier-selected `evidence` citations. The UI shows both, including similarity for retrieval ranking, source metadata, and the method used for that claim.

## Tests

From the repository root, run:

```powershell
python -m pytest backend/tests -q
```

Tests use generated tiny PDFs and fake embedding/LLM interfaces. They cover separate evidence retrieval for multiple claims, relevant and irrelevant passages, contradiction, missing evidence, malformed verifier output, out-of-set IDs, and API error propagation without making paid calls. The first dependency install may still download ML libraries, but tests do not download a sentence embedding model.

## Evaluation

For a same-question comparison, run the evaluation script with `--pipeline baseline` and then `--pipeline verirag`. The baseline retrieves question-level context and generates an answer without claim extraction or verification; VeriRAG adds independent claim retrieval and assessment. Run either pipeline with `--verifier heuristic` or `--verifier llm` (the latter requires a configured key).

1. Create the sample PDF from its readable source: `python backend/scripts/make_sample_pdf.py`.
2. Start VeriRAG and upload `data/sample_documents/sample_policy.pdf` in the browser.
3. From the repository root, run `python backend/scripts/run_evaluation.py --pipeline baseline --verifier heuristic --output data/evaluation/baseline.jsonl`, then rerun with `--pipeline verirag --output data/evaluation/verirag.jsonl` so both outputs are retained.
4. The script writes raw per-question JSONL to `data/evaluation/results.jsonl` and aggregate metrics to `data/evaluation/results_summary.json`. Run with `--verifier llm` to use the evidence-conditioned LLM verifier; live LLM mode requires credentials.

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

### Offline demonstration

The interface offers **Deterministic demo** for the bundled sample policy, and selects it automatically when `/api/health` reports that no LLM key is configured. No provider is called in demo mode. The response is marked `DETERMINISTIC DEMO · NOT LLM OUTPUT`. Try these questions:

- “What does the retention section say?” — directly supported.
- “How long are files kept and are they encrypted at rest?” — one supported claim and one claim with insufficient evidence.
- “Which cloud region stores these documents?” — missing information.
- “What does the conflicting retention note say?” — presents the retention section, the later draft, and the unresolved replacement status.

The bundled fixture is `data/sample_documents/sample_policy.pdf`; upload it from the document panel before asking. The Greenfield Community Library file was not present in this checkout.

See [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md) for beginner-friendly explanations of embeddings, cosine similarity, chunking, RAG, faithfulness, claim extraction, verification, and evaluation.

## Known limitations and next steps

- Scanned PDFs have no extractable text; OCR is not included.
- A local FAISS index and metadata files are appropriate for this prototype, not concurrent multi-user use or a production database.
- Prompt injection defenses instruct the model to treat excerpts as untrusted data, but prompt instructions alone cannot guarantee safe behavior.
- Similarity thresholds are not calibrated truth probabilities and do not establish factual entailment. LLM verification can be wrong or return invalid output; invalid output and provider failures surface as errors.
- Claim-level top-k retrieval can miss relevant passages. `UNSUPPORTED` and `INSUFFICIENT_EVIDENCE` depend on the available documents, retrieval quality, and verifier judgment; neither is a universal truth label.
- There is no claim that this implementation is novel or that verification improves reliability. Establish annotations and run controlled comparisons before making research claims.
- Next research step: refine the question “Does claim-level evidence verification improve the reliability of RAG-generated answers compared with a conventional RAG baseline?” and define the annotation policy before evaluating it.

