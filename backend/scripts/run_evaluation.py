"""Call a running VeriRAG API and store raw rows for later analysis."""
import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.evaluation import evaluate_record, summarize_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/evaluation/questions.jsonl")
    parser.add_argument("--output", default="data/evaluation/results.jsonl")
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/ask")
    parser.add_argument("--verifier", choices=["heuristic", "llm"], default="heuristic")
    args = parser.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with Path(args.dataset).open(encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as output:
        for line in source:
            item = json.loads(line)
            row = {"question_id": item["question_id"], "question": item["question"],
                   "reference_answer": item.get("reference_answer"),
                   "reference_evidence": item.get("reference_evidence", []),
                   "expected_claim_labels": item.get("expected_claim_labels")}
            try:
                request = Request(args.api, data=json.dumps({"question": item["question"],
                    "verifier": args.verifier}).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=180) as response:
                    row.update(json.loads(response.read()))
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                row["error"] = str(exc)
            row["metrics"] = evaluate_record(row)
            records.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{item['question_id']}: {'error' if row.get('error') else 'recorded'}")
    summary = summarize_evaluation(records)
    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

