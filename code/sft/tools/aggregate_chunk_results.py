#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple


def find_json_files(input_dir: str, recursive: bool) -> List[str]:
    paths: List[str] = []
    if recursive:
        for root, dirs, files in os.walk(input_dir):
            # Skip logs dir if present
            dirs[:] = [d for d in dirs if d != "logs"]
            for name in files:
                if name.lower().endswith(".json"):
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            if name.lower().endswith(".json"):
                paths.append(os.path.join(input_dir, name))
    return sorted(paths)


def safe_load_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate chunked inference results and compute per-dataset accuracy."
    )
    parser.add_argument("input_dir", help="Directory containing chunk result JSON files.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for JSON files (skips logs/).",
    )
    parser.add_argument(
        "--output-merged",
        default=None,
        help="Optional path to write merged list JSON.",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Optional path to write summary JSON.",
    )
    args = parser.parse_args()

    json_files = find_json_files(args.input_dir, args.recursive)
    if not json_files:
        raise SystemExit(f"No JSON files found in {args.input_dir}")

    merged: List[dict] = []
    stats: Dict[str, List[int]] = {}
    files_ok = 0

    for path in json_files:
        try:
            items = safe_load_list(path)
        except Exception as e:
            print(f"[WARN] Skip {path}: {e}")
            continue
        files_ok += 1
        for item in items:
            if not isinstance(item, dict):
                continue
            merged.append(item)
            dataset = item.get("dataset") or "UNKNOWN"
            correct = bool(item.get("correct", False))
            if dataset not in stats:
                stats[dataset] = [0, 0]
            if correct:
                stats[dataset][0] += 1
            stats[dataset][1] += 1

    total_correct = sum(v[0] for v in stats.values())
    total_count = sum(v[1] for v in stats.values())

    # Print summary
    print("====================================================================")
    print("Per-dataset accuracy")
    print("====================================================================")
    print(f"Files processed: {files_ok} / {len(json_files)}")
    print(f"Merged items: {len(merged)}")
    print("")
    print(f"{'Dataset':45s} {'Correct/Total':12s} {'Accuracy':10s}")
    print("-" * 68)
    for dataset in sorted(stats.keys()):
        correct, total = stats[dataset]
        acc = (correct / total * 100) if total else 0.0
        print(f"{dataset:45s} {correct:5d} / {total:<5d} {acc:9.2f}%")
    print("-" * 68)
    overall_acc = (total_correct / total_count * 100) if total_count else 0.0
    print(f"{'OVERALL':45s} {total_correct:5d} / {total_count:<5d} {overall_acc:9.2f}%")
    print("====================================================================")

    if args.output_merged:
        with open(args.output_merged, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    if args.output_summary:
        summary = {
            "files_total": len(json_files),
            "files_ok": files_ok,
            "items_total": len(merged),
            "overall": {
                "correct": total_correct,
                "total": total_count,
                "accuracy": round(overall_acc, 2),
            },
            "datasets": {},
        }
        for dataset in sorted(stats.keys()):
            correct, total = stats[dataset]
            acc = (correct / total * 100) if total else 0.0
            summary["datasets"][dataset] = {
                "correct": correct,
                "total": total,
                "accuracy": round(acc, 2),
            }
        with open(args.output_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
