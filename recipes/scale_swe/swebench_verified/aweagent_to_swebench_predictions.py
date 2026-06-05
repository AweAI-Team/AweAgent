#!/usr/bin/env python3
"""Convert AweAgent trajectory JSONL into SWE-bench predictions JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="Scale-SWE-Agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = Path(args.trajectories)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    empty = 0
    with trajectories.open() as src, output.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            patch = record.get("patch") or ""
            if not patch.strip():
                empty += 1
            dst.write(json.dumps({
                "instance_id": record["instance_id"],
                "model_patch": patch,
                "model_name_or_path": args.model_name,
            }, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} predictions to {output} ({empty} empty patches)")


if __name__ == "__main__":
    main()
