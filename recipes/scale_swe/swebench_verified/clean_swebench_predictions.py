#!/usr/bin/env python3
"""Clean obvious non-submission artifacts from SWE-bench predictions JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
TEMP_SCRIPT_RE = re.compile(
    r"(^|/)(debug|try_fix|repro|reproduce|tmp|verify|test)[^/]*\.py$",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input predictions JSONL")
    parser.add_argument("--output", required=True, help="Output predictions JSONL")
    parser.add_argument(
        "--preserve-nonempty-fallback",
        action="store_true",
        help="If cleaning makes a non-empty patch empty, keep the original patch so the harness runs tests.",
    )
    return parser.parse_args()


def split_diff_blocks(patch: str) -> list[str]:
    starts = [match.start() for match in DIFF_HEADER_RE.finditer(patch)]
    if not starts:
        return [patch] if patch.strip() else []
    starts.append(len(patch))
    return [patch[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def block_paths(block: str) -> tuple[str, str] | None:
    first = block.splitlines()[0] if block else ""
    match = re.match(r"diff --git a/(.*?) b/(.*?)$", first)
    if not match:
        return None
    return match.group(1), match.group(2)


def should_drop_block(block: str) -> tuple[bool, str | None]:
    paths = block_paths(block)
    if paths is None:
        return False, None
    old_path, new_path = paths

    if old_path == ".gitignore" and new_path == ".gitignore":
        if "AWEAGENT AUTO-GENERATED" in block or "SWALM AUTO-GENERATED" in block:
            return True, "agent_gitignore"

    if "__pycache__/" in new_path or new_path.endswith(".pyc"):
        return True, "bytecode"

    is_new_file = "\nnew file mode " in block
    if is_new_file and TEMP_SCRIPT_RE.search(new_path):
        return True, "new_temp_script"
    if is_new_file and re.search(r"(^|/)tests?/test[^/]*(/|$)", new_path):
        return True, "new_temp_script"

    return False, None


def clean_patch(patch: str, counts: Counter[str]) -> str:
    kept: list[str] = []
    for block in split_diff_blocks(patch):
        drop, reason = should_drop_block(block)
        if drop:
            counts[reason or "dropped"] += 1
            continue
        kept.append(block)
    return "".join(kept).strip() + ("\n" if kept else "")


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    rows = 0
    changed_rows = 0
    empty_before = 0
    empty_after = 0

    with src.open() as inp, dst.open("w") as out:
        for line in inp:
            if not line.strip():
                continue
            record = json.loads(line)
            old_patch = record.get("model_patch") or ""
            if not old_patch.strip():
                empty_before += 1
            new_patch = clean_patch(old_patch, counts)
            if (
                args.preserve_nonempty_fallback
                and old_patch.strip()
                and not new_patch.strip()
            ):
                new_patch = old_patch
                counts["preserved_nonempty_fallback"] += 1
            if new_patch != old_patch:
                changed_rows += 1
            if not new_patch.strip():
                empty_after += 1
            record["model_patch"] = new_patch
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1

    summary = {
        "rows": rows,
        "changed_rows": changed_rows,
        "empty_before": empty_before,
        "empty_after": empty_after,
        "dropped_blocks": dict(counts),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
