"""Analyze BeyondSWE results — per-task statistics.

For Doc2Repo tasks (domain):
  - Average pass rate
  - Almost correct count (pass_rate >= 0.9)
  - Correct count (pass_rate == 1.0, i.e. all tests passed)

For other tasks (CrossRepo, DomainFix, DepMigrate):
  - Solved % (score == 1.0)

Note: error instances already have score=0.0 in results.jsonl, so they
are correctly counted as failures in all rate calculations.

Usage:
    python recipes/beyond_swe/analyze_results.py --result-dir <output_run_dir>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


DOMAIN_TASK = "doc2repo"  # The task type with partial pass rates


def load_results(result_dir: str) -> list[dict]:
    """Load results.jsonl from the run directory."""
    path = os.path.join(result_dir, "results.jsonl")
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def analyze(results: list[dict]) -> dict:
    """Compute per-task statistics."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    unknown = []
    for r in results:
        task = r.get("task", "").lower()
        if not task:
            unknown.append(r["instance_id"])
            continue
        by_task[task].append(r)

    stats = {}

    for task_type in sorted(by_task.keys()):
        task_results = by_task[task_type]
        total = len(task_results)

        if task_type == DOMAIN_TASK:
            scores = [r["score"] for r in task_results]
            avg_pass_rate = sum(scores) / total if total > 0 else 0.0
            almost_correct = sum(1 for s in scores if s >= 0.9)
            correct = sum(1 for s in scores if s >= 1.0)
            error_count = sum(1 for r in task_results if r.get("finish_reason") == "error")

            stats[task_type] = {
                "total": total,
                "avg_pass_rate": round(avg_pass_rate, 4),
                "almost_correct_num": almost_correct,
                "almost_correct_rate": round(almost_correct / total, 4) if total > 0 else 0.0,
                "correct_num": correct,
                "correct_rate": round(correct / total, 4) if total > 0 else 0.0,
                "error_count": error_count,
            }
        else:
            solved = sum(1 for r in task_results if r["score"] >= 1.0)
            error_count = sum(1 for r in task_results if r.get("finish_reason") == "error")

            stats[task_type] = {
                "total": total,
                "solved": solved,
                "solved_rate": round(solved / total, 4) if total > 0 else 0.0,
                "error_count": error_count,
            }

    # Overall
    total_all = len(results)
    solved_all = sum(1 for r in results if r["score"] >= 1.0)
    error_all = sum(1 for r in results if r.get("finish_reason") == "error")
    stats["overall"] = {
        "total": total_all,
        "solved": solved_all,
        "solved_rate": round(solved_all / total_all, 4) if total_all > 0 else 0.0,
        "error_count": error_all,
    }

    if unknown:
        stats["unknown_instances"] = unknown

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze BeyondSWE results")
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Path to the run output directory (containing results.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: <result-dir>/analysis.json)",
    )
    args = parser.parse_args()

    results = load_results(args.result_dir)
    print(f"Loaded {len(results)} results")

    stats = analyze(results)

    # Print summary to console
    for task_type, s in sorted(stats.items()):
        if task_type == "unknown_instances":
            continue
        if task_type == DOMAIN_TASK:
            print(f"  {task_type:12s}  avg_pass_rate={s['avg_pass_rate']:.2%}  "
                  f"almost_correct={s['almost_correct_num']}/{s['total']}  "
                  f"correct={s['correct_num']}/{s['total']}  "
                  f"errors={s['error_count']}")
        elif task_type == "overall":
            print(f"  {'Overall':12s}  solved={s['solved']}/{s['total']} ({s['solved_rate']:.2%})  "
                  f"errors={s['error_count']}")
        else:
            print(f"  {task_type:12s}  solved={s['solved']}/{s['total']} ({s['solved_rate']:.2%})  "
                  f"errors={s['error_count']}")

    if "unknown_instances" in stats:
        print(f"\n  WARNING: {len(stats['unknown_instances'])} instances missing 'task' field")

    # Save JSON
    output_path = args.output or os.path.join(args.result_dir, "analysis.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
