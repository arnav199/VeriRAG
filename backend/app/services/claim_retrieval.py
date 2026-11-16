"""Retrieve evidence independently for each extracted answer claim."""

from backend.app.schemas import EvidenceResult


class ClaimEvidenceRetrievalService:
    """Embed one claim at a time and search the shared FAISS index for its evidence."""

    def __init__(self, embedder, store, default_top_k: int):
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k

    def retrieve(self, claim: str, top_k: int | None = None) -> list[EvidenceResult]:
        vector = self.embedder.encode([claim])[0]
        return self.store.search(vector, top_k or self.default_top_k)

