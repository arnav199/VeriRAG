"""Deterministic scripted provider for the bundled demonstration policy.

This provider never calls a language model. It exists to make the UI workflow
repeatable when no API key or network service is available.
"""
import json


class DeterministicDemoProvider:
    is_demo = True
    verifier_method = "deterministic_demo_fixture"

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user)
        if system.startswith("Answer the user's question"):
            return self._answer(payload)
        if system.startswith("Extract concise"):
            return {"claims": self._claims(payload["answer"])}
        if system.startswith("Assess only"):
            return self._verify(payload["claim"], payload["evidence"])
        raise ValueError("Unknown deterministic demo operation")

    @staticmethod
    def _answer(payload: dict) -> dict:
        question = payload["question"].casefold()
        evidence = payload["evidence"]
        corpus = " ".join(item["text"].casefold() for item in evidence)
        if "uploaded research documents" not in corpus and "stored locally" not in corpus:
            return {"answer": "The bundled sample policy is not present in the retrieved passages. "
                              "Upload data/sample_documents/sample_policy.pdf and ask again.",
                    "source_chunk_ids": []}
        if "cloud region" in question or "which region" in question:
            return {"answer": "The sample policy does not specify a cloud region.",
                    "source_chunk_ids": []}
        if "encrypt" in question:
            answer = ("The policy says uploaded research documents are stored locally and kept until "
                      "the researcher deletes them. It does not state whether local files are "
                      "encrypted at rest.")
            return {"answer": answer, "source_chunk_ids": DeterministicDemoProvider._ids(evidence)}
        if "conflict" in question or "draft note" in question or "thirty days" in question:
            answer = ("The retention section says documents are kept until the researcher deletes "
                      "them. A later draft note says records are removed after thirty days. "
                      "The document does not explain whether that note replaces the retention section.")
            return {"answer": answer, "source_chunk_ids": DeterministicDemoProvider._ids(evidence)}
        answer = ("The policy says uploaded research documents are stored locally and kept until "
                  "the researcher deletes them.")
        return {"answer": answer, "source_chunk_ids": DeterministicDemoProvider._ids(evidence)}

    @staticmethod
    def _ids(evidence: list[dict]) -> list[str]:
        return list(dict.fromkeys(item["chunk_id"] for item in evidence))

    @staticmethod
    def _claims(answer: str) -> list[str]:
        if answer.startswith("The bundled sample policy is not present"):
            return []
        return [part.strip() + ("." if not part.endswith(".") else "")
                for part in answer.split(". ") if part.strip()]

    @staticmethod
    def _verify(claim: str, evidence: list[dict]) -> dict:
        text = " ".join(item["text"].casefold() for item in evidence)
        lowered = claim.casefold()
        if "cloud region" in lowered or "encrypted at rest" in lowered:
            return {"label": "INSUFFICIENT_EVIDENCE", "evidence_chunk_ids": [],
                    "explanation": "The deterministic demo fixture has no evidence that answers this claim."}
        if "kept until" in lowered and "kept until" in text:
            ids = [item["chunk_id"] for item in evidence if "kept until" in item["text"].casefold()]
        elif "later draft note" in lowered and "thirty days" in text:
            ids = [item["chunk_id"] for item in evidence if "thirty days" in item["text"].casefold()]
        elif "does not explain whether" in lowered and "does not explain" in text:
            ids = [item["chunk_id"] for item in evidence if "does not explain" in item["text"].casefold()]
        else:
            ids = []
        if ids:
            return {"label": "SUPPORTED", "evidence_chunk_ids": list(dict.fromkeys(ids)),
                    "explanation": "The scripted demo fixture found matching wording in the supplied claim-specific passage."}
        return {"label": "INSUFFICIENT_EVIDENCE", "evidence_chunk_ids": [],
                "explanation": "The deterministic demo fixture cannot assess this claim from the retrieved passage."}
