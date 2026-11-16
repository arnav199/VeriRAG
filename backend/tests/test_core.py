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
from backend.app.services.claim_retrieval import ClaimEvidenceRetrievalService
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


def test_faiss_index_persists_and_reloads_searchable_vectors(tmp_path):
    pytest.importorskip("faiss")
    from backend.app.schemas import DocumentRecord
    record = DocumentRecord(document_id="persisted", filename="persisted.pdf", sha256="abc",
                            page_count=1, chunk_count=1, uploaded_at="2026-01-01T00:00:00+00:00")
    chunk = ChunkRecord(chunk_id="persisted-p1-c1", document_id="persisted",
                         filename="persisted.pdf", page_number=1, text="persistent evidence")
    store = LocalVectorStore(tmp_path)
    store.add_document(record, [chunk], np.array([[1.0, 0.0]], dtype="float32"))

    reloaded = LocalVectorStore(tmp_path)
    results = reloaded.search(np.array([1.0, 0.0]), 1)
    assert len(results) == 1
    assert results[0].chunk_id == "persisted-p1-c1"
    assert results[0].similarity == pytest.approx(1.0)


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


def test_invalid_provider_key_has_useful_error_without_exposing_key(monkeypatch):
    from urllib.error import HTTPError
    import backend.app.services.llm as llm_module

    secret = "must-not-be-printed"
    def unauthorized(*args, **kwargs):
        raise HTTPError("https://provider.invalid", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(llm_module, "urlopen", unauthorized)
    with pytest.raises(LLMError, match="rejected the configured API key") as raised:
        OpenAICompatibleProvider(secret, "https://provider.invalid/v1", "model").complete_json("s", "u")
    assert secret not in str(raised.value)


def test_claim_retrieval_embeds_each_claim_and_preserves_source_metadata():
    alpha = EvidenceResult(chunk_id="alpha-p2-c1", document_id="alpha", filename="alpha.pdf",
                           page_number=2, text="Alpha fact.", similarity=.87)
    beta = EvidenceResult(chunk_id="beta-p5-c1", document_id="beta", filename="beta.pdf",
                          page_number=5, text="Beta fact.", similarity=.79)

    class RoutingEmbedder:
        def __init__(self): self.queries = []
        def encode(self, texts):
            self.queries.extend(texts)
            return np.array([[1, 0] if "alpha" in texts[0].lower() else [0, 1]], dtype="float32")

    class RoutingStore:
        def search(self, vector, top_k):
            assert top_k == 4
            return [alpha] if vector[0] else [beta]

    embedder = RoutingEmbedder()
    service = ClaimEvidenceRetrievalService(embedder, RoutingStore(), default_top_k=4)
    alpha_result = service.retrieve("Claim about alpha")
    beta_result = service.retrieve("Claim about beta")
    assert embedder.queries == ["Claim about alpha", "Claim about beta"]
    assert alpha_result[0].chunk_id == "alpha-p2-c1" and alpha_result[0].similarity == .87
    assert beta_result[0].filename == "beta.pdf" and beta_result[0].page_number == 5


def test_llm_verifier_handles_contradictory_and_irrelevant_evidence():
    contradiction = EvidenceResult(chunk_id="c1", document_id="d", filename="policy.pdf", page_number=3,
                                  text="The policy says records expire after 30 days.", similarity=.84)
    contradicted = llm_verify("Records are kept forever.", [contradiction], FakeLLM([{
        "label": "CONTRADICTED", "evidence_chunk_ids": ["c1"],
        "explanation": "The passage states a 30-day expiration.",
    }]))
    assert contradicted.label == VerificationLabel.CONTRADICTED
    assert contradicted.evidence[0].chunk_id == "c1"
    assert contradicted.retrieved_evidence[0].similarity == .84

    irrelevant = EvidenceResult(chunk_id="c2", document_id="x", filename="unrelated.pdf", page_number=1,
                                text="A sentence about gardening.", similarity=.21)
    abstained = llm_verify("The capital is Paris.", [irrelevant], FakeLLM([{
        "label": "INSUFFICIENT_EVIDENCE", "evidence_chunk_ids": [],
        "explanation": "The retrieved passage does not address the claim.",
    }]))
    assert abstained.label == VerificationLabel.INSUFFICIENT_EVIDENCE
    assert abstained.retrieved_evidence == [irrelevant]
    assert abstained.evidence == []


@pytest.mark.parametrize("label", ["PARTIALLY_SUPPORTED", "UNSUPPORTED"])
def test_llm_verifier_accepts_partial_and_unsupported_labels(label):
    evidence = EvidenceResult(chunk_id="c1", document_id="d", filename="policy.pdf", page_number=2,
                              text="The policy mentions some but not all requested conditions.", similarity=.74)
    result = llm_verify("A multi-part claim.", [evidence], FakeLLM([{
        "label": label, "evidence_chunk_ids": ["c1"], "explanation": "Assessment based on the supplied passage.",
    }]))
    assert result.label.value == label
    assert result.evidence[0].chunk_id == "c1"


def test_claim_verifier_missing_evidence_does_not_call_llm():
    result = llm_verify("A factual claim.", [], FakeLLM([]))
    assert result.label == VerificationLabel.INSUFFICIENT_EVIDENCE
    assert result.retrieved_evidence == []
    assert result.evidence == []


@pytest.mark.parametrize("response", [
    {"label": "SUPPORTED", "explanation": "missing ids"},
    {"label": "SUPPORTED", "evidence_chunk_ids": ["c1"], "explanation": "  "},
    {"label": "SUPPORTED", "evidence_chunk_ids": [], "explanation": "no evidence cited"},
    {"label": "PARTIALLY_SUPPORTED", "evidence_chunk_ids": ["not-retrieved"], "explanation": "bad id"},
    {"label": "UNKNOWN", "evidence_chunk_ids": ["c1"], "explanation": "bad label"},
])
def test_claim_verifier_rejects_malformed_output_and_invalid_evidence_ids(response):
    evidence = EvidenceResult(chunk_id="c1", document_id="d", filename="doc.pdf", page_number=1,
                              text="Evidence.", similarity=.7)
    with pytest.raises(LLMError):
        llm_verify("Claim.", [evidence], FakeLLM([response]))


def test_multiple_claims_get_different_retrieval_and_verification_evidence():
    answer_evidence = EvidenceResult(chunk_id="answer", document_id="answer-doc", filename="answer.pdf",
                                     page_number=1, text="Answer context.", similarity=.65)
    cats = EvidenceResult(chunk_id="cats-p4-c1", document_id="cats", filename="cats.pdf",
                          page_number=4, text="Cats are mammals.", similarity=.88)
    oceans = EvidenceResult(chunk_id="oceans-p8-c1", document_id="oceans", filename="oceans.pdf",
                            page_number=8, text="The Pacific is an ocean.", similarity=.81)

    class ClaimEmbedder:
        def encode(self, texts):
            text = texts[0].lower()
            if "cats" in text: vector = [1, 0, 0]
            elif "pacific" in text: vector = [0, 1, 0]
            else: vector = [0, 0, 1]
            return np.array([vector], dtype="float32")

    class ClaimStore:
        def search(self, vector, top_k):
            if vector[0]: return [cats]
            if vector[1]: return [oceans]
            return [answer_evidence]

    llm = FakeLLM([
        {"answer": "Cats are mammals. The Pacific is an ocean.", "source_chunk_ids": ["answer"]},
        {"claims": ["Cats are mammals.", "The Pacific is an ocean."]},
        {"label": "SUPPORTED", "evidence_chunk_ids": ["cats-p4-c1"], "explanation": "The cats passage states this."},
        {"label": "SUPPORTED", "evidence_chunk_ids": ["oceans-p8-c1"], "explanation": "The oceans passage states this."},
    ])
    result = RAGService(ClaimEmbedder(), ClaimStore(), llm, 3, "llm").ask("Give me two facts", verifier="llm")
    assert [item.chunk_id for item in result.retrieved_evidence] == ["answer"]
    assert [item.retrieved_evidence[0].chunk_id for item in result.claims] == ["cats-p4-c1", "oceans-p8-c1"]
    assert [item.evidence[0].chunk_id for item in result.claims] == ["cats-p4-c1", "oceans-p8-c1"]
    assert result.claims[0].retrieved_evidence[0].filename == "cats.pdf"
    assert result.claims[1].retrieved_evidence[0].page_number == 8


def test_llm_verification_api_error_is_returned_not_labeled(tmp_path):
    evidence = EvidenceResult(chunk_id="c1", document_id="d", filename="doc.pdf", page_number=1,
                              text="A relevant passage.", similarity=.9)
    class Store:
        def search(self, vector, top_k): return [evidence]

    llm = FakeLLM([
        {"answer": "A claim.", "source_chunk_ids": ["c1"]},
        {"claims": ["A claim."]},
        LLMError("provider unavailable"),
    ])
    app = create_app(settings(tmp_path), embedder=FakeEmbedder(), llm=llm, store=Store())
    response = TestClient(app).post("/api/ask", json={"question": "Question", "verifier": "llm"})
    assert response.status_code == 502
    assert "claims" not in response.json()


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


def test_pdf_upload_api_extracts_pages_indexes_and_deduplicates(tmp_path):
    app = create_app(settings(tmp_path), embedder=FakeEmbedder(),
                     llm=OpenAICompatibleProvider(None, "http://localhost/v1", "unused"),
                     store=LocalVectorStore(tmp_path))
    client = TestClient(app)
    pdf = make_pdf("An uploaded PDF contains page-aware evidence for this test.")
    first = client.post("/api/documents", files={"file": ("evidence.pdf", pdf, "application/pdf")})
    assert first.status_code == 201
    result = first.json()
    assert result["duplicate"] is False
    assert result["document"]["page_count"] == 1
    assert result["document"]["chunk_count"] >= 1
    assert client.get("/api/documents").json()[0]["filename"] == "evidence.pdf"

    repeated = client.post("/api/documents", files={"file": ("renamed.pdf", pdf, "application/pdf")})
    assert repeated.status_code == 201
    assert repeated.json()["duplicate"] is True
    assert repeated.json()["document"]["document_id"] == result["document"]["document_id"]


def test_deterministic_demo_covers_supported_partial_missing_and_conflict(tmp_path):
    from backend.app.services.demo import DeterministicDemoProvider
    policy = EvidenceResult(
        chunk_id="policy-p1-c1", document_id="policy", filename="sample_policy.pdf",
        page_number=1,
        text=("Uploaded research documents are stored locally and kept until the researcher deletes them. "
              "The sample policy does not state whether local files are encrypted at rest. "
              "A later draft note says records are removed after thirty days. "
              "The note does not explain whether it replaces the retention section."), similarity=.9)

    class Store:
        def search(self, vector, top_k): return [policy]

    service = RAGService(FakeEmbedder(), Store(), FakeLLM([]), 3, "heuristic")
    supported = service.ask("What does the retention section say?", mode="demo")
    assert supported.generation_mode == "deterministic_demo"
    assert supported.claims[0].label == VerificationLabel.SUPPORTED
    assert supported.claims[0].verifier_method == "deterministic_demo_fixture"

    partial = service.ask("How long are files kept and are they encrypted at rest?", mode="demo")
    assert [claim.label for claim in partial.claims] == [
        VerificationLabel.SUPPORTED, VerificationLabel.INSUFFICIENT_EVIDENCE]

    missing = service.ask("Which cloud region stores these documents?", mode="demo")
    assert missing.claims[0].label == VerificationLabel.INSUFFICIENT_EVIDENCE
    assert missing.claims[0].evidence == []

    conflict = service.ask("What does the conflicting retention note say?", mode="demo")
    assert len(conflict.claims) == 3
    assert all(claim.label == VerificationLabel.SUPPORTED for claim in conflict.claims)
    assert "thirty days" in conflict.answer
    assert "DEMO MODE" in " ".join(conflict.limitations)

    app = create_app(settings(tmp_path), embedder=FakeEmbedder(), llm=FakeLLM([]), store=Store())
    response = TestClient(app).post("/api/ask", json={
        "question": "What does the retention section say?", "mode": "demo",
    })
    assert response.status_code == 200
    assert response.json()["generation_mode"] == "deterministic_demo"


def test_baseline_skips_claim_extraction_and_verification():
    evidence = EvidenceResult(chunk_id="c1", document_id="d", filename="doc.pdf", page_number=1,
                              text="A sample evidence passage.", similarity=.8)
    class Store:
        def search(self, vector, top_k): return [evidence]
    result = RAGService(FakeEmbedder(), Store(), FakeLLM([
        {"answer": "An answer based on evidence.", "source_chunk_ids": ["c1"]},
    ]), 2, "heuristic").ask("Question?", pipeline="baseline")
    assert result.pipeline == "baseline"
    assert result.claims == []
    assert result.verifier_method == "conventional_rag_baseline"


def test_demo_does_not_generate_fixture_facts_without_the_sample_policy():
    class EmptyStore:
        def search(self, vector, top_k): return []
    result = RAGService(FakeEmbedder(), EmptyStore(), FakeLLM([]), 3, "heuristic").ask(
        "What does the retention section say?", mode="demo")
    assert "not present in the retrieved passages" in result.answer
    assert result.citations == [] and result.claims == []

