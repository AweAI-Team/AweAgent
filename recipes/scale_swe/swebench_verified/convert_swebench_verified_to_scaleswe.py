#!/usr/bin/env python3
"""Convert SWE-bench Verified rows into AweAgent ScaleSWE JSONL.

The converter keeps the fields AweAgent needs for inference and preserves the
official SWE-bench fields needed later by the harness.  By default it chooses
instances whose official SWE-bench instance image is already present locally,
which is useful for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--instance-ids", nargs="*", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-local-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--namespace", default="swebench")
    return parser.parse_args()


def docker_images() -> set[str]:
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        check=True,
        text=True,
        capture_output=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def split_repo(repo: str) -> tuple[str, str]:
    if "/" not in repo:
        return "", repo
    owner, name = repo.split("/", 1)
    return owner, name


def to_scaleswe(row: dict[str, Any], image: str) -> dict[str, Any]:
    owner, repo_name = split_repo(row["repo"])
    base_commit = row["base_commit"]
    return {
        "instance_id": row["instance_id"],
        "user": owner,
        "repo": repo_name,
        "parent_commit": base_commit,
        "base_commit": base_commit,
        "image_url": image,
        "image": image,
        "workdir": "/testbed",
        "language": "python",
        "problem_statement": row["problem_statement"],
        "pre_commands": (
            f"git checkout {base_commit} -f && "
            f"git reset --hard {base_commit} && "
            "git clean -fdx"
        ),
        "test_patch": row.get("test_patch", ""),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", "[]"),
        "PASS_TO_PASS": row.get("PASS_TO_PASS", "[]"),
        "patch": row.get("patch", ""),
        "version": row.get("version", ""),
    }


def main() -> None:
    args = parse_args()
    local_images = docker_images() if args.require_local_image else set()
    wanted = set(args.instance_ids or [])
    dataset = load_dataset(args.dataset_name, split=args.split)

    rows: list[dict[str, Any]] = []
    for raw in dataset:
        row = dict(raw)
        if wanted and row["instance_id"] not in wanted:
            continue
        image = make_test_spec(row, namespace=args.namespace).instance_image_key
        if args.require_local_image and image not in local_images:
            continue
        rows.append(to_scaleswe(row, image))

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    rows = rows[:args.limit]

    if not rows:
        raise SystemExit("No matching instances found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {output}")
    for row in rows:
        print(f"{row['instance_id']} image={row['image_url']}")


if __name__ == "__main__":
    main()
