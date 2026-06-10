#!/usr/bin/env python3
"""Materialize UniReason-Med 2D/3D mixed RL data for verl.

The public HF dataset stores RL records in the `rl` split and 2D image bytes in
the separate `images` split. This script resolves 2D `image_ids` to relative
local image paths, keeps 3D examples text-only, and writes train/validation
Parquet files with verl-native `reward_model` and `extra_info` dictionaries.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _safe_image_name(path_value: str, image_id: str) -> str:
    raw = Path(path_value or f"{image_id}.png")
    return raw.name or f"{image_id}.png"


def _collect_needed_image_ids(rl_files: Iterable[Path], batch_size: int) -> set[str]:
    needed: set[str] = set()
    for file in rl_files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(columns=["image_ids"], batch_size=batch_size):
            for ids in batch.column(0).to_pylist():
                needed.update(ids or [])
    return needed


def _write_needed_images(
    image_files: Iterable[Path], needed_ids: set[str], output_dir: Path, path_root: Path, batch_size: int
) -> dict[str, str]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    id_to_path: dict[str, str] = {}

    for file in image_files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(columns=["image_id", "path", "bytes"], batch_size=batch_size):
            for row in batch.to_pylist():
                image_id = row["image_id"]
                if image_id not in needed_ids:
                    continue
                out_path = image_dir / _safe_image_name(row.get("path") or "", image_id)
                if not out_path.exists():
                    out_path.write_bytes(row["bytes"])
                id_to_path[image_id] = Path(os.path.relpath(out_path.resolve(), path_root.resolve())).as_posix()
    missing = needed_ids.difference(id_to_path)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"Missing {len(missing)} released images. Examples: {preview}")
    return id_to_path


def _loads_json_dict(value, default=None):
    if value is None:
        return {} if default is None else default
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Expected dict or JSON string, got {type(value).__name__}")


def _iter_rows(rl_files: Iterable[Path], id_to_path: dict[str, str], batch_size: int):
    for file in rl_files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                image_ids = row.get("image_ids") or []
                yield {
                    "data_source": row.get("data_source") or row.get("source_dataset") or "unireason_med",
                    "agent_name": row.get("agent_name") or "urm_single_assistant",
                    "prompt": row["prompt"],
                    "images": [id_to_path[image_id] for image_id in image_ids],
                    "ability": row.get("ability") or "medical_vqa",
                    "reward_model": _loads_json_dict(row.get("reward_model"), default={"style": "rule"}),
                    "extra_info": _loads_json_dict(row.get("extra_info"), default={}),
                }


def _split_rows(rows, val_size: int):
    index = 0
    val_buffer = []
    train_buffer = []
    for row in rows:
        if index < val_size:
            val_buffer.append(row)
        else:
            train_buffer.append(row)
        index += 1
        if len(val_buffer) >= 4096:
            yield "val", val_buffer
            val_buffer = []
        if len(train_buffer) >= 4096:
            yield "train", train_buffer
            train_buffer = []
    if val_buffer:
        yield "val", val_buffer
    if train_buffer:
        yield "train", train_buffer


def _resolve_source(args: argparse.Namespace) -> Path:
    if args.source_dir:
        return Path(args.source_dir).expanduser().resolve()
    return Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=["rl/*.parquet", "images/*.parquet"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir", default=None, help="Use an existing dataset snapshot instead of downloading.")
    parser.add_argument("--output-dir", default="data/unireason_med_rl")
    parser.add_argument("--path-root", default=".", help="Root used to make image paths relative in the output parquet.")
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    source_dir = _resolve_source(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    path_root = Path(args.path_root).expanduser()

    rl_files = _iter_parquet_files(source_dir, "rl")
    image_files = _iter_parquet_files(source_dir, "images")
    needed_ids = _collect_needed_image_ids(rl_files, args.batch_size)
    id_to_path = _write_needed_images(image_files, needed_ids, output_dir, path_root, args.batch_size)

    train_file = output_dir / "train.parquet"
    val_file = output_dir / "val.parquet"
    train_writer = None
    val_writer = None
    train_count = 0
    val_count = 0
    try:
        for split, chunk in _split_rows(_iter_rows(rl_files, id_to_path, args.batch_size), args.val_size):
            table = pa.Table.from_pylist(chunk)
            if split == "train":
                if train_writer is None:
                    train_writer = pq.ParquetWriter(train_file, table.schema, compression="zstd")
                train_writer.write_table(table)
                train_count += len(chunk)
            else:
                if val_writer is None:
                    val_writer = pq.ParquetWriter(val_file, table.schema, compression="zstd")
                val_writer.write_table(table)
                val_count += len(chunk)
    finally:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

    print(f"Wrote {train_count} train rows to {train_file}")
    print(f"Wrote {val_count} validation rows to {val_file}")
    print(f"Wrote {len(id_to_path)} images under {output_dir / 'images'}")


if __name__ == "__main__":
    main()
