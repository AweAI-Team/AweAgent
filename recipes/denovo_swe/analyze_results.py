"""Analyze DeNovoSWE evaluation results.

Reads results.jsonl from a run directory and computes statistics:
- avg_pass_rate, correct_num/rate, almost_correct (>=0.8), error_count

Usage:
    python recipes/denovo_swe/analyze_results.py results/denovoswe/run_XXXX/results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# error_kind values that denote infrastructure failures (excluded from the
# scored denominator). See aweagent.core.task.types.ErrorKind.
_INFRA_KINDS = {"infra_error", "timeout", "context_length"}


def analyze(results_file: str) -> dict:
    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        return {"error": "no results"}

    total = len(results)
    scores = []
    correct = 0
    almost_correct = 0
    errors = 0

    for r in results:
        eval_result = r.get("eval_result", {})
        if not eval_result:
            errors += 1
            continue

        score = eval_result.get("score", 0.0)
        accepted = eval_result.get("accepted", False)
        details = eval_result.get("details", {})

        # Prefer the structured error_kind; fall back to the details["error"]
        # heuristic for trajectories written before error_kind existed.
        kind = eval_result.get("error_kind")
        is_infra = kind in _INFRA_KINDS if kind is not None else bool(details.get("error"))
        if is_infra:
            errors += 1
            continue

        scores.append(score)
        if accepted:
            correct += 1
        if score >= 0.8:
            almost_correct += 1

    n_scored = len(scores)
    avg_pass_rate = sum(scores) / n_scored if n_scored > 0 else 0.0

    analysis = {
        "total": total,
        "scored": n_scored,
        "errors": errors,
        "correct_num": correct,
        "correct_rate": correct / total if total > 0 else 0.0,
        "almost_correct_num": almost_correct,
        "almost_correct_rate": almost_correct / total if total > 0 else 0.0,
        "avg_pass_rate": avg_pass_rate,
    }

    return analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results.jsonl>")
        sys.exit(1)

    results_file = sys.argv[1]
    analysis = analyze(results_file)

    # Save alongside results
    output_path = Path(results_file).parent / "analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(json.dumps(analysis, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
