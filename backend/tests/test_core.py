import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.evaluation import (answer_token_f1, citation_support_rate, classification_metrics,
                                    evaluate_record, retrieval_metrics, summarize_evaluation,
                                    summarize_records)
from backend.app.main import create_app
from backend.app.schemas import ChunkRecord, EvidenceResult, VerificationLabel
from backend.app.services.chunking import chunk_pages
from backend.app.services.documents import extract_pdf_pages
from backend.app.services.llm import LLMError, LLMNotConfigured, OpenAICompatibleProvider
from backend.app.services.rag import RAGService
from backend.app.services.storage import LocalVectorStore
from backend.app.services.verification import extract_claims, heuristic_verify, llm_verify


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0, float(len(text) % 2)] for text in texts], dtype="float32")


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete_json(self, system, user):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def settings(tmp_path):
    return Settings(tmp_path, "unused", 120, 10, 3, 1024 * 1024, "heuristic",
                    None, "http://localhost/v1", "unused")


def make_pdf(text="VeriRAG stores facts with page references."):
    import pymupdf
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def test_pdf_extraction_page_metadata_and_chunking():
    pages = extract_pdf_pages(make_pdf("Page one evidence for the study."))
    assert pages[0][0] == 1 and "Page one evidence" in pages[0][1]
    chunks = chunk_pages(pages, "doc", "paper.pdf", chunk_size=15, overlap=3)
    assert chunks and all(c.page_number == 1 and c.filename == "paper.pdf" for c in chunks)
    assert chunks[0].document_id == "doc" and chunks[0].chunk_id


def test_bad_and_textless_pdf_errors():
    with pytest.raises(ValueError):
        extract_pdf_pages(b"not a pdf")
    import pymupdf
    pdf = pymupdf.open(); pdf.new_page(); empty = pdf.tobytes(); pdf.close()
    with pytest.raises(ValueError, match="No extractable text"):
        from backend.app.services.documents import DocumentService
        class EmptyStore:
            def find_by_hash(self, digest): return None
        DocumentService(EmptyStore(), FakeEmbedder(), 120, 10).ingest("empty.pdf", empty)


def test_document_duplicate_and_persistent_metadata(tmp_path, monkeypatch):
    import backend.app.services.storage as storage
    class FakeFaissIndex:
        def __init__(self, d): self.d, self.v = d, []
        def add(self, rows): self.v.extend(rows.tolist())
        def search(self, q, k):
            sims = np.array(self.v) @ q[0]
            ids = np.argsort(sims)[::-1][:k]
            return sims[ids][None, :], ids[None, :]
    class FakeFaiss:
        IndexFlatIP = FakeFaissIndex
        @staticmethod
        def write_index(index, path): open(path, "wb").close()
        @staticmethod
        def read_index(path): return FakeFaissIndex(2)
    monkeypatch.setitem(__import__("sys").modules, "faiss", FakeFaiss)
    from backend.app.services.documents import DocumentService
    store = LocalVectorStore(tmp_path)
    service = DocumentService(store, FakeEmbedder(), 120, 10)
    pdf_bytes = make_pdf()
    first, duplicate = service.ingest("facts.pdf", pdf_bytes)
    again, is_duplicate = service.ingest("different-name.pdf", pdf_bytes)
    assert not duplicate and is_duplicate and first.document_id == again.document_id
    assert (tmp_path / "documents.json").exists() and store.find_by_hash(first.sha256)


def test_embedding_and_retrieval_interfaces():
    class Store:
        def search(self, vector, top_k):
            assert len(vector) == 2 and top_k == 2
            return []
    assert RAGService(FakeEmbedder(), Store(), FakeLLM([]), 3, "heuristic").retrieve("question", 2) == []


def test_rag_response_schema_keeps_answer_evidence_and_claims():
    evidence = EvidenceResult(chunk_id="c1", document_id="d", filename="paper.pdf", page_number=1,
                              text="Retention lasts until the owner deletes the file.", similarity=.82)
    class Store:
        def search(self, vector, top_k): return [evidence]
    result = RAGService(FakeEmbedder(), Store(), FakeLLM([
        {"answer": "Files are retained until their owner deletes them.", "source_chunk_ids": ["c1"]},
        {"claims": ["Files are retained until their owner deletes them."]},
    ]), 3, "heuristic").ask("How long are files kept?")
    assert result.answer.startswith("Files are retained")
    assert result.citations[0].page_number == 1
    assert result.claims[0].label == VerificationLabel.SUPPORTED


def test_claim_parsing_verification_and_missing_evidence():
    llm = FakeLLM([{"claims": ["The system stores page numbers."]}])
    assert extract_claims("It stores page numbers.", llm) == ["The system stores page numbers."]
    no_evidence = heuristic_verify("claim", [])
    assert no_evidence.label == VerificationLabel.INSUFFICIENT_EVIDENCE
    evidence = EvidenceResult(chunk_id="c1", document_id="d", filename="a.pdf", page_number=2,
                              text="The system stores page numbers.", similarity=.8)
    result = heuristic_verify("claim", [evidence])
    assert result.label == VerificationLabel.SUPPORTED and "not proof" in result.explanation
    result = llm_verify("claim", [evidence], FakeLLM([{"label": "CONTRADICTED",
        "evidence_chunk_ids": ["c1"], "explanation": "The passage says otherwise."}]))
    assert result.label == VerificationLabel.CONTRADICTED


def test_malformed_llm_outputs_and_missing_key():
    with pytest.raises(LLMError):
        extract_claims("answer", FakeLLM([{"claims": "not-json"}]))
    with pytest.raises(LLMError):
        llm_verify("claim", [EvidenceResult(chunk_id="c", document_id="d", filename="x", page_number=1,
          text="evidence", similarity=.9)], FakeLLM([{"label": "INVALID", "explanation": "no", "evidence_chunk_ids": []}]))
    with pytest.raises(LLMNotConfigured):
        OpenAICompatibleProvider(None, "http://localhost/v1", "model").complete_json("system", "user")


def test_evaluation_metrics():
    result = classification_metrics(["SUPPORTED", "UNSUPPORTED", "SUPPORTED"],
                                    ["SUPPORTED", "SUPPORTED", "UNSUPPORTED"])
    assert result == {"positive_label": "SUPPORTED", "precision": .5, "recall": .5, "f1": .5, "n": 3}
    summary = summarize_records([{"claims": [{"label": "UNSUPPORTED"}], "latency_ms": 100}])
    assert summary["unsupported_claim_rate"] == 1 and summary["mean_latency_ms"] == 100
    assert answer_token_f1("red green", "green red") == 1
    assert retrieval_metrics(["a", "b"], ["b", "c"]) == {"precision": .5, "recall": .5, "n_relevant": 2}
    assert citation_support_rate(["a", "b"], ["a"]) == .5
    assert evaluate_record({"answer": "hello world", "reference_answer": "hello", "claims": [],
                            "retrieved_evidence": [], "citations": []})["claim_verification"] is None
    summary = summarize_evaluation([{"answer": "hello world", "reference_answer": "hello",
                                     "claims": [], "retrieved_evidence": [], "citations": []}])
    assert summary["mean_answer_token_f1"] == pytest.approx(2 / 3)


def test_api_validation_and_upload_errors(tmp_path):
    app = create_app(settings(tmp_path), embedder=FakeEmbedder(),
                     llm=OpenAICompatibleProvider(None, "http://localhost/v1", "unused"))
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/retrieval", json={"query": ""}).status_code == 422
    assert client.post("/api/documents", files={"file": ("notes.txt", b"hello")}).status_code == 415
    assert client.post("/api/documents", files={"file": ("bad.pdf", b"not a pdf")}).status_code == 422
    assert client.post("/api/documents", files={"file": ("empty.pdf", b"")}).status_code == 422
    assert client.post("/api/documents", files={"file": ("large.pdf", b"%PDF-" + b"x" * (1024 * 1024))}).status_code == 413
    ask = client.post("/api/ask", json={"question": "A valid question"})
    assert ask.status_code == 503 and "LLM_API_KEY" in ask.json()["detail"]

