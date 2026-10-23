import json
import re

from backend.app.schemas import Citation, ClaimVerification, EvidenceResult, VerificationLabel
from backend.app.services.llm import LLMError


def citations_for(evidence: list[EvidenceResult]) -> list[Citation]:
    return [Citation(chunk_id=e.chunk_id, document_id=e.document_id, filename=e.filename,
                     page_number=e.page_number, text=e.text) for e in evidence]


def _parse_json_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("Expected a JSON array")


def extract_claims(answer: str, llm) -> list[str]:
    result = llm.complete_json(
        "Extract concise, independently verifiable factual claims from the answer. "
        "Return JSON as {\"claims\":[\"...\"]}. Do not add facts.",
        json.dumps({"answer": answer}),
    )
    try:
        claims = _parse_json_list(result.get("claims"))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError("The claim extractor returned malformed JSON.") from exc
    if not all(isinstance(item, str) and item.strip() for item in claims):
        raise LLMError("The claim extractor returned invalid claim entries.")
    return [item.strip() for item in claims]


def heuristic_verify(claim: str, evidence: list[EvidenceResult]) -> ClaimVerification:
    if not evidence:
        return ClaimVerification(claim=claim, label=VerificationLabel.INSUFFICIENT_EVIDENCE,
            verifier_method="cosine_similarity_heuristic", explanation="No retrieved evidence was available.")
    best = max(evidence, key=lambda item: item.similarity)
    score = max(-1.0, min(1.0, best.similarity))
    # These cutoffs rank topical similarity only; they do not establish factual entailment.
    label = (VerificationLabel.SUPPORTED if score >= 0.72 else
             VerificationLabel.PARTIALLY_SUPPORTED if score >= 0.48 else
             VerificationLabel.INSUFFICIENT_EVIDENCE)
    return ClaimVerification(claim=claim, label=label, evidence=citations_for([best]),
        verifier_method="cosine_similarity_heuristic", raw_similarity=score,
        explanation="Similarity is a topical heuristic, not proof that the evidence entails this claim.")


def llm_verify(claim: str, evidence: list[EvidenceResult], llm) -> ClaimVerification:
    if not evidence:
        return ClaimVerification(claim=claim, label=VerificationLabel.INSUFFICIENT_EVIDENCE,
            verifier_method="evidence_conditioned_llm", explanation="No retrieved evidence was available.")
    raw = llm.complete_json(
        "Assess only the supplied claim and evidence. Return JSON with label (SUPPORTED, "
        "PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE), "
        "evidence_chunk_ids (array of supplied IDs), and explanation. Similarity is not proof. "
        "Use INSUFFICIENT_EVIDENCE when evidence does not decide the claim.",
        json.dumps({"claim": claim, "evidence": [e.model_dump() for e in evidence]}, ensure_ascii=False),
    )
    try:
        label = VerificationLabel(raw["label"])
        explanation = raw["explanation"]
        ids = raw.get("evidence_chunk_ids", [])
        if not isinstance(explanation, str) or not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError("invalid verifier fields")
        by_id = {item.chunk_id: item for item in evidence}
        if any(item_id not in by_id for item_id in ids):
            raise ValueError("verifier referenced evidence outside the retrieved set")
    except (KeyError, ValueError, TypeError) as exc:
        raise LLMError("The verifier returned invalid structured output.") from exc
    used = [by_id[item_id] for item_id in ids]
    return ClaimVerification(claim=claim, label=label, evidence=citations_for(used),
        verifier_method="evidence_conditioned_llm", explanation=explanation,
        raw_similarity=max((item.similarity for item in used), default=None))

