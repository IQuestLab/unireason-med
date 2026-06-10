"""
Reward function for UniReason-Med MCQ GRPO training.

Provides two components:
1. answer_reward: correctness of MCQ option (A/B/C/D)
2. format_reward: structural format quality (<think> pairing and bbox tag pairing)
"""

from __future__ import annotations

import re
from typing import Any, Optional

CHOICE_PATTERN = re.compile(r"\b([A-D])\b", flags=re.IGNORECASE)
THINK_BEGIN_PATTERN = re.compile(r"<think>", flags=re.IGNORECASE)
THINK_END_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)

BOX_START_TOKEN = "<|box_start|>"
BOX_END_TOKEN = "<|box_end|>"


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)


def extract_choice(text: str) -> str:
    if not text:
        return ""
    plain_text = _strip_think(text)
    match = CHOICE_PATTERN.search(plain_text)
    if not match:
        return ""
    return match.group(1).upper()


def compute_format_reward(solution_str: str) -> float:
    if not solution_str:
        return 0.0

    think_begin = len(THINK_BEGIN_PATTERN.findall(solution_str))
    think_end = len(THINK_END_PATTERN.findall(solution_str))
    box_begin = solution_str.count(BOX_START_TOKEN)
    box_end = solution_str.count(BOX_END_TOKEN)

    score = 0.0
    if think_begin == think_end:
        score += 0.5
    if box_begin == box_end:
        score += 0.5
    return score


def compute_answer_reward(solution_str: str, ground_truth: str) -> tuple[float, str, str]:
    pred_choice = extract_choice(solution_str)
    gt_choice = extract_choice(ground_truth)
    if not gt_choice:
        return 0.0, pred_choice, gt_choice
    return (1.0 if pred_choice == gt_choice else 0.0), pred_choice, gt_choice


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    *,
    answer_weight: float = 0.8,
    format_weight: float = 0.2,
    **kwargs,
) -> dict[str, Any]:
    """verl-compatible reward function entry."""
    del data_source, kwargs

    format_reward = compute_format_reward(solution_str)
    answer_reward, pred_choice, gt_choice = compute_answer_reward(solution_str, ground_truth)
    score = answer_weight * answer_reward + format_weight * format_reward

    crop_area_ratio = float((extra_info or {}).get("urm_crop_area_ratio", 0.0) or 0.0)

    return {
        "score": float(score),
        "answer_reward": float(answer_reward),
        "format_reward": float(format_reward),
        "crop_area_ratio": crop_area_ratio,
        "pred_choice": pred_choice,
        "gt_choice": gt_choice,
    }


main = compute_score
