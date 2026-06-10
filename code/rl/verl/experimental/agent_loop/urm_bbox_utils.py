# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random as _random
import re
from typing import Optional

from PIL import Image

BOX_PATTERN_SQUARE = re.compile(r"<\|box_start\|>\[([^\]]+)\]<\|box_end\|>")
BOX_PATTERN_ROUND = re.compile(r"<\|box_start\|>\(([^)]+)\)<\|box_end\|>")


def _parse_coord_str(coord_str: str) -> Optional[list[float]]:
    try:
        coords = [float(x.strip()) for x in coord_str.split(",")]
    except ValueError:
        return None
    return coords


def extract_last_2d_bbox(text: str) -> Optional[list[float]]:
    """Extract the last 2D bbox from text. Supports [x1,y1,x2,y2] and (x1,y1,x2,y2)."""
    if not text:
        return None

    boxes: list[list[float]] = []
    for pattern in (BOX_PATTERN_SQUARE, BOX_PATTERN_ROUND):
        for match in pattern.finditer(text):
            coords = _parse_coord_str(match.group(1))
            if coords and len(coords) == 4:
                boxes.append(coords)

    if not boxes:
        return None
    return boxes[-1]


def generate_random_bbox(
    image: Image.Image,
    pred_bbox: list[float],
    min_size_fraction: float = 0.1,
    max_size_fraction: float = 0.5,
    max_attempts: int = 20,
) -> list[float]:
    """Generate a random bbox with low IoU relative to pred_bbox.

    Samples multiple candidate bboxes and returns the one with minimum IoU
    with pred_bbox. Falls back to the top-left quarter if all attempts fail.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        hw, hh = max(4, width // 2), max(4, height // 2)
        return [0.0, 0.0, float(hw), float(hh)]

    px1, py1, px2, py2 = [float(c) for c in pred_bbox[:4]]

    min_w = max(4, int(min_size_fraction * width))
    max_w = max(min_w + 1, int(max_size_fraction * width))
    min_h = max(4, int(min_size_fraction * height))
    max_h = max(min_h + 1, int(max_size_fraction * height))

    def _iou(bx1, by1, bx2, by2, cx1, cy1, cx2, cy2):
        ix1 = max(bx1, cx1)
        iy1 = max(by1, cy1)
        ix2 = min(bx2, cx2)
        iy2 = min(by2, cy2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        area_c = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        union = area_b + area_c - inter
        return inter / union if union > 0 else 0.0

    best_bbox: Optional[list[float]] = None
    best_iou = float("inf")

    for _ in range(max_attempts):
        w = _random.randint(min_w, max_w)
        h = _random.randint(min_h, max_h)
        x1 = _random.randint(0, max(0, width - w))
        y1 = _random.randint(0, max(0, height - h))
        x2 = x1 + w
        y2 = y1 + h

        iou = _iou(px1, py1, px2, py2, x1, y1, x2, y2)
        if iou < best_iou:
            best_iou = iou
            best_bbox = [float(x1), float(y1), float(x2), float(y2)]

        if iou == 0.0:
            break  # Found a non-overlapping bbox; stop early

    if best_bbox is None:
        # Fallback: top-left quarter
        hw, hh = max(4, width // 2), max(4, height // 2)
        best_bbox = [0.0, 0.0, float(hw), float(hh)]

    return best_bbox


def _expand_interval(start: int, end: int, target_len: int, max_len: int) -> tuple[int, int]:
    """Expand [start, end) to target_len while staying in [0, max_len]."""
    curr_len = end - start
    if target_len <= curr_len:
        return start, end

    add = target_len - curr_len
    left_add = add // 2
    right_add = add - left_add
    new_start = start - left_add
    new_end = end + right_add

    if new_start < 0:
        new_end = min(max_len, new_end - new_start)
        new_start = 0
    if new_end > max_len:
        overflow = new_end - max_len
        new_start = max(0, new_start - overflow)
        new_end = max_len

    if new_end - new_start < target_len:
        if new_start == 0:
            new_end = min(max_len, target_len)
        elif new_end == max_len:
            new_start = max(0, max_len - target_len)

    if new_end <= new_start:
        new_end = min(max_len, new_start + 1)
        if new_end <= new_start:
            new_start = max(0, new_end - 1)
    return new_start, new_end


def crop_2d_image(
    image: Image.Image,
    box: list[float],
    *,
    min_side: int = 4,
    max_aspect_ratio: float = 180.0,
) -> Image.Image:
    """Crop 2D image with bbox [x1, y1, x2, y2], auto-expanding over-thin regions."""
    x1, y1, x2, y2 = [int(c) for c in box[:4]]
    width, height = image.size

    # Base clamp.
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    # Ensure minimum side lengths.
    if min_side > 1:
        x1, x2 = _expand_interval(x1, x2, min(min_side, width), width)
        y1, y2 = _expand_interval(y1, y2, min(min_side, height), height)

    crop_w = x2 - x1
    crop_h = y2 - y1

    # Expand short side if aspect ratio is too extreme.
    if max_aspect_ratio > 1.0 and crop_w > 0 and crop_h > 0:
        ratio = max(crop_w, crop_h) / min(crop_w, crop_h)
        if ratio > max_aspect_ratio:
            if crop_w >= crop_h:
                target_h = min(height, int(math.ceil(crop_w / max_aspect_ratio)))
                y1, y2 = _expand_interval(y1, y2, max(1, target_h), height)
            else:
                target_w = min(width, int(math.ceil(crop_h / max_aspect_ratio)))
                x1, x2 = _expand_interval(x1, x2, max(1, target_w), width)

    crop_w = x2 - x1
    crop_h = y2 - y1

    # Final safety fallback: if still extreme (rare), use full image.
    if crop_w <= 0 or crop_h <= 0:
        return image
    if max_aspect_ratio > 1.0 and max(crop_w, crop_h) / min(crop_w, crop_h) > max_aspect_ratio:
        return image

    return image.crop((x1, y1, x2, y2))


def _smart_resize_like_qwen(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """
    Qwen-like smart resize:
    - Round to multiples of factor
    - If too big: scale down
    - If too small: scale up
    Returns (new_h, new_w)
    """
    if height < factor or width < factor:
        h_bar = max(factor, math.ceil(height / factor) * factor)
        w_bar = max(factor, math.ceil(width / factor) * factor)
        return h_bar, w_bar

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    num_pixels = h_bar * w_bar

    if num_pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif num_pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    h_bar = max(factor, int(h_bar))
    w_bar = max(factor, int(w_bar))
    return h_bar, w_bar


def resize_pil_to_processor(img: Image.Image, processor) -> Image.Image:
    """
    Resize PIL image to processor size using Qwen smart-resize.
    This keeps bbox coordinates aligned with processor-scale image.
    """
    image_processor = getattr(processor, "image_processor", None)
    patch = getattr(image_processor, "patch_size", 14) if image_processor is not None else 14
    merge = getattr(image_processor, "merge_size", 2) if image_processor is not None else 2
    factor = patch * merge

    min_pixels = getattr(image_processor, "min_pixels", None) if image_processor is not None else None
    if min_pixels is None:
        min_pixels = 3136

    max_pixels = getattr(image_processor, "max_pixels", None) if image_processor is not None else None
    if max_pixels is None:
        max_pixels = 12845056

    orig_w, orig_h = img.size
    proc_h, proc_w = _smart_resize_like_qwen(
        orig_h,
        orig_w,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if (proc_w, proc_h) != (orig_w, orig_h):
        return img.resize((proc_w, proc_h), Image.BILINEAR)
    return img
