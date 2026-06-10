#!/usr/bin/env python3
"""Materialize UniReason-Med SFT release data for LLaMA-Factory.

The public HF dataset stores training rows in the `sft` split and image bytes
in the separate `images` split. LLaMA-Factory expects an `images` column with
local paths, so this script writes image files and creates a compact Parquet
file with `messages` and `images` columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "IQuestLab/UniReason-Med-Data"


def _iter_parquet_files(root: Path, split: str) -> list[Path]:
    files = sorted((root / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / split}")
    return files


def _safe_relative_image_path(path_value: str, image_id: str) -> Path:
    raw = Path(path_value or f"images/{image_id}.png")
    name = raw.name or f"{image_id}.png"
    return Path("images") / name


def _collect_needed_image_ids(sft_files: Iterable[Path], batch_size: int) -> set[str]:
    needed: set[str] = set()
    for file in sft_files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(columns=["image_ids"], batch_size=batch_size):
            for ids in batch.column(0).to_pylist():
                needed.update(ids or [])
    return needed


def _write_needed_images(image_files: Iterable[Path], needed_ids: set[str], output_dir: Path, batch_size: int) -> dict[str, str]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    id_to_path: dict[str, str] = {}

    for file in image_files:
        parquet_file = pq.ParquetFile(file)
        columns = ["image_id", "path", "bytes"]
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
            for row in batch.to_pylist():
                image_id = row["image_id"]
                if image_id not in needed_ids:
                    continue
                rel_path = _safe_relative_image_path(row.get("path") or "", image_id)
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not out_path.exists():
                    out_path.write_bytes(row["bytes"])
                id_to_path[image_id] = rel_path.as_posix()
    missing = needed_ids.difference(id_to_path)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"Missing {len(missing)} released images. Examples: {preview}")
    return id_to_path


def _write_training_parquet(sft_files: Iterable[Path], id_to_path: dict[str, str], output_file: Path, batch_size: int) -> int:
    schema = pa.schema(
        [
            pa.field("messages", pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())]))),
            pa.field("images", pa.list_(pa.string())),
        ]
    )
    count = 0
    with pq.ParquetWriter(output_file, schema=schema, compression="zstd") as writer:
        for file in sft_files:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(columns=["messages", "image_ids"], batch_size=batch_size):
                rows = []
                for row in batch.to_pylist():
                    image_ids = row.get("image_ids") or []
                    rows.append(
                        {
                            "messages": row["messages"],
                            "images": [id_to_path[image_id] for image_id in image_ids],
                        }
                    )
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(rows)
    return count


def _resolve_source(args: argparse.Namespace) -> Path:
    if args.source_dir:
        return Path(args.source_dir).expanduser().resolve()
    return Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=["sft/*.parquet", "images/*.parquet"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir", default=None, help="Use an existing dataset snapshot instead of downloading.")
    parser.add_argument("--output-dir", default="data/unireason_med_sft")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    source_dir = _resolve_source(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_files = _iter_parquet_files(source_dir, "sft")
    image_files = _iter_parquet_files(source_dir, "images")
    needed_ids = _collect_needed_image_ids(sft_files, args.batch_size)
    id_to_path = _write_needed_images(image_files, needed_ids, output_dir, args.batch_size)
    row_count = _write_training_parquet(sft_files, id_to_path, output_dir / "train.parquet", args.batch_size)

    print(f"Wrote {row_count} SFT rows to {output_dir / 'train.parquet'}")
    print(f"Wrote {len(id_to_path)} images under {output_dir / 'images'}")


if __name__ == "__main__":
    main()
