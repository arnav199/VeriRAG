from enum import Enum
from pydantic import BaseModel, Field


class VerificationLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTED = "CONTRADICTED"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    sha256: str
    page_count: int
    chunk_count: int
    uploaded_at: str


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str


class EvidenceResult(ChunkRecord):
    similarity: float


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str


class ClaimVerification(BaseModel):
    claim: str
    label: VerificationLabel
    evidence: list[Citation] = Field(default_factory=list)
    verifier_method: str
    explanation: str
    raw_similarity: float | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    verifier: str | None = Field(default=None, pattern="^(heuristic|llm)$")


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved_evidence: list[EvidenceResult]
    claims: list[ClaimVerification]
    verifier_method: str
    latency_ms: float
    limitations: list[str]


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvaluationRecord(BaseModel):
    question_id: str
    question: str
    generated_answer: str | None = None
    claims: list[ClaimVerification] = Field(default_factory=list)
    reference_evidence: list[str] = Field(default_factory=list)
    retrieval_results: list[EvidenceResult] = Field(default_factory=list)
    latency_ms: float | None = None
    verifier_method: str | None = None
    error: str | None = None

