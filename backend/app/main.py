from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings
from backend.app.schemas import AskRequest, RetrievalRequest
from backend.app.services.documents import DocumentService
from backend.app.services.embedding import SentenceTransformerEmbedder
from backend.app.services.llm import LLMError, LLMNotConfigured, OpenAICompatibleProvider
from backend.app.services.rag import RAGService
from backend.app.services.storage import LocalVectorStore


def create_app(settings: Settings | None = None, embedder=None, llm=None, store=None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or LocalVectorStore(settings.data_dir)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    llm = llm or OpenAICompatibleProvider(
        settings.llm_api_key, settings.llm_base_url, settings.llm_model
    )
    documents = DocumentService(store, embedder, settings.chunk_size, settings.chunk_overlap)
    rag = RAGService(embedder, store, llm, settings.top_k, settings.verifier)

    app = FastAPI(title="VeriRAG", version="0.1.0", description=
                  "Local document RAG with claim-level evidence assessments.")
    app.state.settings, app.state.store = settings, store
    app.state.documents, app.state.rag = documents, rag
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "llm_configured": bool(settings.llm_api_key),
                "document_count": len(store.documents)}

    @app.get("/api/documents")
    def list_documents():
        return store.documents

    @app.post("/api/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...)):
        filename = (file.filename or "").strip()
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(415, "Only PDF files are accepted.")
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, f"PDF exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.")
        if not content:
            raise HTTPException(422, "The uploaded file is empty.")
        if not content.startswith(b"%PDF-"):
            raise HTTPException(422, "The uploaded file does not have a PDF signature.")
        try:
            record, duplicate = documents.ingest(filename, content)
            return {"document": record, "duplicate": duplicate}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "Document processing is unavailable. Check the backend setup and local model.") from exc

    @app.post("/api/retrieval")
    def inspect_retrieval(request: RetrievalRequest):
        try:
            return {"query": request.query, "results": rag.retrieve(request.query, request.top_k)}
        except Exception as exc:
            raise HTTPException(503, "Retrieval is unavailable. Check the embedding model and local index.") from exc

    @app.post("/api/ask")
    def ask(request: AskRequest):
        try:
            return rag.ask(request.question, request.top_k, request.verifier)
        except LLMNotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "RAG is unavailable. Check the embedding model, index, and configuration.") from exc

    return app


app = create_app()

