#!/usr/bin/env python3
"""Convert a verl FSDP checkpoint (sharded .pt) into a HuggingFace model.

This is a thin wrapper around scripts/legacy_model_merger.py for convenience.

python scripts/convert_fsdp_to_hf.py \
    --checkpoint_dir checkpoints/unireason_med_grpo/global_step_260 \
    --target_dir checkpoints/unireason_med_grpo/global_step_260/actor/merged_hf
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _resolve_actor_dir(checkpoint_dir: str) -> str:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if os.path.basename(checkpoint_dir) == "actor":
        return checkpoint_dir
    candidate = os.path.join(checkpoint_dir, "actor")
    if os.path.isdir(candidate):
        return candidate
    return checkpoint_dir


def _ensure_fsdp_files(actor_dir: str) -> None:
    pattern = os.path.join(actor_dir, "model_world_size_*_rank_0.pt")
    if not glob.glob(pattern):
        raise FileNotFoundError(
            f"No FSDP shard files found under {actor_dir}. Expected pattern: {pattern}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert verl FSDP checkpoint to HuggingFace format",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to global_step_xx or actor dir",
    )
    parser.add_argument(
        "--target_dir",
        required=True,
        help="Output directory for merged HuggingFace model",
    )
    args = parser.parse_args()

    actor_dir = _resolve_actor_dir(args.checkpoint_dir)
    _ensure_fsdp_files(actor_dir)

    # Import merger utilities from scripts/legacy_model_merger.py
    sys.path.append(os.path.dirname(__file__))
    from legacy_model_merger import FSDPModelMerger, ModelMergerConfig  # noqa: E402

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=actor_dir,
        hf_model_config_path=actor_dir,
        target_dir=os.path.abspath(args.target_dir),
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()


if __name__ == "__main__":
    main()
