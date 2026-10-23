"""Metric helpers use only rows with the relevant human labels or references."""
from collections import Counter
import re


def classification_metrics(predicted: list[str], expected: list[str], positive_label: str = "SUPPORTED") -> dict:
    if len(predicted) != len(expected):
        raise ValueError("Predicted and expected labels must have equal length")
    tp = sum(p == positive_label and y == positive_label for p, y in zip(predicted, expected))
    fp = sum(p == positive_label and y != positive_label for p, y in zip(predicted, expected))
    fn = sum(p != positive_label and y == positive_label for p, y in zip(predicted, expected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"positive_label": positive_label, "precision": precision, "recall": recall, "f1": f1,
            "n": len(expected)}


def summarize_records(records: list[dict]) -> dict:
    labels = Counter(claim.get("label") for row in records for claim in row.get("claims", []))
    claim_count = sum(labels.values())
    unsupported = labels.get("UNSUPPORTED", 0) + labels.get("CONTRADICTED", 0)
    latencies = [row["latency_ms"] for row in records if isinstance(row.get("latency_ms"), (int, float))]
    return {
        "question_count": len(records), "claim_count": claim_count,
        "label_counts": dict(labels),
        "unsupported_claim_rate": unsupported / claim_count if claim_count else None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "note": "Unsupported-claim rate is dataset- and verifier-dependent; it is not a universal hallucination rate.",
    }


def answer_token_f1(prediction: str, reference: str) -> float:
    """Token overlap F1; useful as a rough text match, not a semantic correctness judge."""
    tokenize = lambda text: re.findall(r"\w+", text.casefold())
    predicted, expected = Counter(tokenize(prediction)), Counter(tokenize(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected:
        return float(predicted == expected)
    precision, recall = overlap / sum(predicted.values()), overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: list[str]) -> dict:
    if not relevant_ids:
        return {"precision": None, "recall": None, "n_relevant": 0,
                "note": "No relevance annotations were supplied."}
    relevant = set(relevant_ids)
    found = sum(item_id in relevant for item_id in retrieved_ids)
    return {"precision": found / len(retrieved_ids) if retrieved_ids else 0.0,
            "recall": found / len(relevant), "n_relevant": len(relevant)}


def citation_support_rate(cited_ids: list[str], relevant_ids: list[str]) -> float | None:
    if not relevant_ids or not cited_ids:
        return None
    relevant = set(relevant_ids)
    return sum(item_id in relevant for item_id in cited_ids) / len(cited_ids)


def evaluate_record(row: dict) -> dict:
    """Return only metrics supported by supplied references and annotations."""
    reference = row.get("reference_answer")
    actual = row.get("answer", row.get("generated_answer"))
    relevant_ids = row.get("reference_evidence_ids") or []
    retrieved = row.get("retrieved_evidence", row.get("retrieval_results", []))
    citations = row.get("citations", [])
    labels = row.get("expected_claim_labels")
    predictions = [claim.get("label") for claim in row.get("claims", [])]
    return {
        "answer_token_f1": answer_token_f1(actual, reference) if isinstance(actual, str) and isinstance(reference, str) else None,
        "claim_verification": classification_metrics(predictions, labels) if isinstance(labels, list) and len(labels) == len(predictions) else None,
        "retrieval": retrieval_metrics([item.get("chunk_id") for item in retrieved], relevant_ids),
        "citation_support_rate": citation_support_rate([item.get("chunk_id") for item in citations], relevant_ids),
    }


def summarize_evaluation(records: list[dict]) -> dict:
    summary = summarize_records(records)
    per_record = [evaluate_record(row) for row in records]
    for metric_name, target_name in [
        ("answer_token_f1", "mean_answer_token_f1"),
        ("citation_support_rate", "mean_citation_support_rate"),
    ]:
        values = [item[metric_name] for item in per_record if item[metric_name] is not None]
        summary[target_name] = sum(values) / len(values) if values else None
    for metric in ("precision", "recall"):
        values = [item["retrieval"][metric] for item in per_record
                  if item["retrieval"][metric] is not None]
        summary[f"mean_retrieval_{metric}"] = sum(values) / len(values) if values else None
    predicted, expected = [], []
    for row in records:
        labels = row.get("expected_claim_labels")
        claims = row.get("claims", [])
        if isinstance(labels, list) and len(labels) == len(claims):
            expected.extend(labels)
            predicted.extend(claim.get("label") for claim in claims)
    summary["supported_claim_classification"] = (
        classification_metrics(predicted, expected) if expected else None
    )
    return summary

