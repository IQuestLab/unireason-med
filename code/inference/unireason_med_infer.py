#!/usr/bin/env python3
"""UniReason-Med Hugging Face inference with 2D and 3D grounding crops.

This script is extracted from the original UniReason-Med evaluation code and
keeps only the inference path:
- 2D images: generate until <|box_end|>, crop [x1,y1,x2,y2], and continue.
- 3D volumes: serialize a volume into slices, crop [x1,y1,z1,x2,y2,z2], and continue.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_PATH = "IQuestLab/UniReason-Med"
DEFAULT_3D_SIZE = (32, 256, 256)

BOX_PATTERN = re.compile(r"<\|box_start\|>[\[\(]([^\]\)]+)[\]\)]<\|box_end\|>")

SYSTEM_PROMPT_2D = (
    "You are a medical expert skilled in analyzing 2D medical images. "
    "You are capable of interleaved image-text thinking. In the <think></think> tag, "
    "you can use <|box_start|>[x1,y1,x2,y2]<|box_end|> to focus on that region of the image. "
    "You can inspect it once or multiple times and think in an interleaved manner to answer the question. "
    "You need to answer a single-choice question. You only need to reply with option_letter. option_content."
)

SYSTEM_PROMPT_3D = (
    "You are a medical expert skilled in analyzing 3D medical images. "
    "You are capable of interleaved image-text thinking. In the <think></think> tag, "
    "you can use <|box_start|>[x1,y1,z1,x2,y2,z2]<|box_end|> to focus on that region of the image. "
    "You can inspect it once or multiple times and think in an interleaved manner to answer the question."
)


def parse_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def first_model_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    for param in model.parameters():
        if not getattr(param, "is_meta", False):
            return param.device
    return fallback


def parse_target_size(value: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target size must be D,H,W")
    if any(x <= 0 for x in parts):
        raise argparse.ArgumentTypeError("target size values must be positive")
    return parts[0], parts[1], parts[2]


def parse_all_bboxes(text: str) -> list[tuple[list[float], bool]]:
    boxes: list[tuple[list[float], bool]] = []
    for match in BOX_PATTERN.finditer(text or ""):
        try:
            coords = [float(x.strip()) for x in match.group(1).split(",")]
        except ValueError:
            continue
        if len(coords) in (4, 6):
            boxes.append((coords, len(coords) == 6))
    return boxes


def extract_answer_choice(response: str) -> Optional[str]:
    if not response:
        return None
    plain = re.sub(r"<think>.*?</think>", " ", response, flags=re.IGNORECASE | re.DOTALL)
    patterns = [
        r"\banswer\s*(?:is|:)?\s*([A-D])\b",
        r"\b([A-D])\.\s*$",
        r"\b([A-D])\s*$",
        r"\b([A-D])\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, plain, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def smart_resize_like_qwen(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
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

    return max(factor, int(h_bar)), max(factor, int(w_bar))


def resize_pil_to_processor(img: Image.Image, processor: Any) -> Image.Image:
    image_processor = getattr(processor, "image_processor", None)
    patch = getattr(image_processor, "patch_size", 14) if image_processor is not None else 14
    merge = getattr(image_processor, "merge_size", 2) if image_processor is not None else 2
    factor = patch * merge

    min_pixels = getattr(image_processor, "min_pixels", None) if image_processor is not None else None
    max_pixels = getattr(image_processor, "max_pixels", None) if image_processor is not None else None
    min_pixels = 3136 if min_pixels is None else min_pixels
    max_pixels = 12845056 if max_pixels is None else max_pixels

    orig_w, orig_h = img.size
    proc_h, proc_w = smart_resize_like_qwen(orig_h, orig_w, factor, min_pixels, max_pixels)
    if (proc_w, proc_h) != (orig_w, orig_h):
        return img.resize((proc_w, proc_h), Image.BILINEAR)
    return img


def expand_interval(start: int, end: int, target_len: int, max_len: int) -> tuple[int, int]:
    current_len = end - start
    if target_len <= current_len:
        return start, end

    add = target_len - current_len
    new_start = start - add // 2
    new_end = end + add - add // 2

    if new_start < 0:
        new_end = min(max_len, new_end - new_start)
        new_start = 0
    if new_end > max_len:
        new_start = max(0, new_start - (new_end - max_len))
        new_end = max_len

    if new_end <= new_start:
        new_end = min(max_len, new_start + 1)
    return new_start, new_end


def crop_2d_image(image: Image.Image, box: list[float], min_side: int = 4, max_aspect_ratio: float = 180.0) -> Image.Image:
    x1, y1, x2, y2 = [int(c) for c in box[:4]]
    width, height = image.size

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    if min_side > 1:
        x1, x2 = expand_interval(x1, x2, min(min_side, width), width)
        y1, y2 = expand_interval(y1, y2, min(min_side, height), height)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return image

    ratio = max(crop_w, crop_h) / min(crop_w, crop_h)
    if max_aspect_ratio > 1.0 and ratio > max_aspect_ratio:
        if crop_w >= crop_h:
            target_h = min(height, int(math.ceil(crop_w / max_aspect_ratio)))
            y1, y2 = expand_interval(y1, y2, max(1, target_h), height)
        else:
            target_w = min(width, int(math.ceil(crop_h / max_aspect_ratio)))
            x1, x2 = expand_interval(x1, x2, max(1, target_w), width)

    return image.crop((x1, y1, x2, y2))


def crop_3d_slices(images: list[Image.Image], box: list[float]) -> list[Image.Image]:
    x1, y1, z1, x2, y2, z2 = [int(c) for c in box[:6]]
    depth = len(images)
    if depth == 0:
        return []
    z1 = max(0, min(z1, depth - 1))
    z2 = max(z1 + 1, min(z2, depth))
    return [crop_2d_image(img, [x1, y1, x2, y2]) for img in images[z1:z2]]


def load_volume(volume_path: str | Path, target_size: tuple[int, int, int], normalize: str = "percentile") -> torch.Tensor:
    path = Path(volume_path)
    if path.suffix == ".npz":
        data = np.load(path)
        first_key = sorted(data.files)[0]
        volume = data[first_key]
    elif path.suffix in {".npy", ""}:
        volume = np.load(path)
    elif path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError("NIfTI input requires nibabel. Install nibabel or convert the volume to .npy.") from exc
        volume = np.asarray(nib.load(str(path)).get_fdata())
    else:
        raise ValueError(f"Unsupported volume format: {path}")

    volume = np.asarray(volume, dtype=np.float32).squeeze()
    if volume.ndim == 4:
        if volume.shape[0] <= 4:
            volume = volume[0]
        elif volume.shape[-1] <= 4:
            volume = volume[..., 0]
        else:
            volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume after squeeze, got shape={volume.shape}")

    depth_axis = min(range(3), key=lambda axis: abs(volume.shape[axis] - target_size[0]))
    volume = np.moveaxis(volume, depth_axis, 0)

    if normalize == "percentile":
        lo, hi = np.percentile(volume, [1.0, 99.0])
        if hi > lo:
            volume = (volume - lo) / (hi - lo)
    elif normalize == "minmax":
        vmin, vmax = float(volume.min()), float(volume.max())
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unsupported normalization: {normalize}")

    volume = np.clip(volume, 0.0, 1.0)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)
    return tensor.squeeze(0).squeeze(0)


def volume_to_slices(volume: torch.Tensor) -> list[Image.Image]:
    if volume.dim() == 4:
        volume = volume[0]
    volume = volume.detach().cpu().float().clamp(0, 1)
    slices: list[Image.Image] = []
    for z in range(volume.shape[0]):
        arr = (volume[z] * 255.0).to(torch.uint8).numpy()
        slices.append(Image.fromarray(arr, mode="L").convert("RGB"))
    return slices


def compact_assistant_delimiter(text: str) -> str:
    delimiter = "<|im_end|>\n<|im_start|>assistant\n"
    first_idx = text.find(delimiter)
    if first_idx < 0:
        return text
    before_first = text[: first_idx + len(delimiter)]
    after_first = text[first_idx + len(delimiter) :].replace(delimiter, "")
    return before_first + after_first


def collect_images(messages: list[dict[str, Any]]) -> list[Any]:
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                images.append(item["image"])
    return images


def decode_assistant_text(processor: Any, generated_ids: torch.Tensor) -> str:
    full_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    assistant_start = "<|im_start|>assistant\n"
    assistant_end = "<|im_end|>"
    start_idx = full_text.find(assistant_start)
    if start_idx < 0:
        return ""
    content_start = start_idx + len(assistant_start)
    end_idx = full_text.rfind(assistant_end)
    if end_idx != -1 and end_idx > content_start:
        return full_text[content_start:end_idx].strip()
    return full_text[content_start:].strip()


class UniReasonMedHFInference:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        quiet: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        dtype = parse_dtype(torch_dtype)
        self.quiet = quiet

        if not self.quiet:
            print(f"Loading processor from {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        resolved_device_map: Optional[str] = None if device_map == "none" else device_map
        if resolved_device_map == "auto" and self.device.type == "cpu":
            resolved_device_map = None
        if resolved_device_map is not None:
            model_kwargs["device_map"] = resolved_device_map

        if not self.quiet:
            print(f"Loading model from {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs).eval()
        if resolved_device_map is None:
            self.model = self.model.to(self.device)
        self.input_device = first_model_device(self.model, self.device)

        if not self.quiet:
            print(f"Model ready on {self.input_device}")

    @torch.inference_mode()
    def generate_2d(
        self,
        question: str,
        image_path: str | Path,
        *,
        max_new_tokens: int = 512,
        max_rounds: int = 10,
        tokens_per_round: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        raw_image = Image.open(image_path).convert("RGB")
        aligned_image = resize_pil_to_processor(raw_image, self.processor)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_2D},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        return self._generate_with_grounding(
            messages=messages,
            crop_images=[aligned_image],
            max_new_tokens=max_new_tokens,
            max_rounds=max_rounds,
            tokens_per_round=tokens_per_round,
            temperature=temperature,
            top_p=top_p,
        )

    @torch.inference_mode()
    def generate_3d(
        self,
        question: str,
        volume_path: str | Path,
        *,
        target_size: tuple[int, int, int] = DEFAULT_3D_SIZE,
        normalize: str = "percentile",
        max_new_tokens: int = 512,
        max_rounds: int = 10,
        tokens_per_round: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        volume = load_volume(volume_path, target_size=target_size, normalize=normalize)
        raw_images = volume_to_slices(volume)
        aligned_images = [resize_pil_to_processor(img, self.processor) for img in raw_images]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_3D},
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in raw_images]
                + [{"type": "text", "text": question}],
            },
        ]
        return self._generate_with_grounding(
            messages=messages,
            crop_images=aligned_images,
            max_new_tokens=max_new_tokens,
            max_rounds=max_rounds,
            tokens_per_round=tokens_per_round,
            temperature=temperature,
            top_p=top_p,
        )

    def _generate_with_grounding(
        self,
        *,
        messages: list[dict[str, Any]],
        crop_images: list[Image.Image],
        max_new_tokens: int,
        max_rounds: int,
        tokens_per_round: int,
        temperature: float,
        top_p: float,
    ) -> str:
        tokenizer = self.processor.tokenizer
        eos_id = tokenizer.eos_token_id
        box_end_id = tokenizer.convert_tokens_to_ids("<|box_end|>")
        if box_end_id is not None and box_end_id < 0:
            box_end_id = None

        stop_ids = [int(x) for x in (eos_id, box_end_id) if x is not None]
        final_generated_ids: Optional[torch.Tensor] = None
        total_generated = 0

        for _ in range(max_rounds):
            remaining_tokens = max_new_tokens - total_generated
            if remaining_tokens <= 0:
                break

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = compact_assistant_delimiter(text)
            image_inputs = collect_images(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
                padding=True,
            ).to(self.input_device)

            do_sample = temperature > 0.0
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": min(tokens_per_round, remaining_tokens),
                "do_sample": do_sample,
            }
            if stop_ids:
                generation_kwargs["eos_token_id"] = stop_ids
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p

            generated_ids = self.model.generate(**inputs, **generation_kwargs)
            final_generated_ids = generated_ids
            generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            if not generated_trimmed or generated_trimmed[0].numel() == 0:
                break

            total_generated += int(generated_trimmed[0].numel())
            response = self.processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            last_token = int(generated_trimmed[0][-1].item())

            stopped_on_box = box_end_id is not None and last_token == int(box_end_id)
            if stopped_on_box:
                boxes = parse_all_bboxes(response)
                if boxes:
                    coords, is_3d = boxes[-1]
                    if is_3d:
                        cropped_images = crop_3d_slices(crop_images, coords)
                    else:
                        cropped_images = [crop_2d_image(crop_images[0], coords)] if crop_images else []
                    if cropped_images:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": response}]
                                + [{"type": "image", "image": img} for img in cropped_images],
                            }
                        )
                        continue

            messages.append({"role": "assistant", "content": response})
            break

        if final_generated_ids is None:
            return ""
        return decode_assistant_text(self.processor, final_generated_ids)


def format_choices(question: str, choices: dict[str, str], *, add_choice_instruction: bool) -> str:
    lines = [question]
    choice_lines = [f"{letter}. {choices[letter]}" for letter in ("A", "B", "C", "D") if choices.get(letter)]
    if choice_lines:
        lines.append("\n".join(choice_lines))
        if add_choice_instruction:
            lines.append("Please answer with A, B, C, or D.")
    return "\n\n".join(lines)


def single_choices_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "A": args.option_a or "",
        "B": args.option_b or "",
        "C": args.option_c or "",
        "D": args.option_d or "",
    }


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_data_path(path_value: str, root: Optional[str]) -> Path:
    path = Path(path_value)
    if path.is_absolute() or not root:
        return path
    return Path(root) / path


def build_engine(args: argparse.Namespace) -> UniReasonMedHFInference:
    return UniReasonMedHFInference(
        model_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        quiet=args.quiet,
    )


def run_2d(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    prompt = format_choices(args.question, single_choices_from_args(args), add_choice_instruction=False)
    response = engine.generate_2d(
        prompt,
        args.image,
        max_new_tokens=args.max_new_tokens,
        max_rounds=args.max_rounds,
        tokens_per_round=args.tokens_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result = {"image": args.image, "question": args.question, "response": response, "prediction": extract_answer_choice(response)}
    if args.output:
        write_json(Path(args.output), result)
    print(response)


def run_3d(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    prompt = format_choices(args.question, single_choices_from_args(args), add_choice_instruction=True)
    response = engine.generate_3d(
        prompt,
        args.volume,
        target_size=args.target_size,
        normalize=args.normalize,
        max_new_tokens=args.max_new_tokens,
        max_rounds=args.max_rounds,
        tokens_per_round=args.tokens_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result = {"volume": args.volume, "question": args.question, "response": response, "prediction": extract_answer_choice(response)}
    if args.output:
        write_json(Path(args.output), result)
    print(response)


def run_2d_batch(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    records = load_json_records(Path(args.input))
    if args.max_samples is not None:
        records = records[: args.max_samples]

    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question = record["question"]
        choices = {
            "A": record.get("option_A", ""),
            "B": record.get("option_B", ""),
            "C": record.get("option_C", ""),
            "D": record.get("option_D", ""),
        }
        prompt = format_choices(question, choices, add_choice_instruction=False)
        image_path = resolve_data_path(record["image_path"], args.image_root)
        try:
            response = engine.generate_2d(
                prompt,
                image_path,
                max_new_tokens=args.max_new_tokens,
                max_rounds=args.max_rounds,
                tokens_per_round=args.tokens_per_round,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output = dict(record)
            output.update({"response": response, "prediction": extract_answer_choice(response), "error": ""})
        except Exception as exc:
            output = dict(record)
            output.update({"response": "", "prediction": None, "error": repr(exc)})
        outputs.append(output)
        if not args.quiet:
            print(f"[2d-batch] {index + 1}/{len(records)} done")

    write_json(Path(args.output), outputs)


def run_3d_batch(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        choices = {letter: row.get(f"Choice {letter}", "") for letter in ("A", "B", "C", "D")}
        prompt = format_choices(row["Question"], choices, add_choice_instruction=True)
        volume_path = resolve_data_path(row["Image Path"], args.data_root)
        try:
            response = engine.generate_3d(
                prompt,
                volume_path,
                target_size=args.target_size,
                normalize=args.normalize,
                max_new_tokens=args.max_new_tokens,
                max_rounds=args.max_rounds,
                tokens_per_round=args.tokens_per_round,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output = dict(row)
            output.update({"response": response, "prediction": extract_answer_choice(response), "error": ""})
        except Exception as exc:
            output = dict(row)
            output.update({"response": "", "prediction": None, "error": repr(exc)})
        outputs.append(output)
        if not args.quiet:
            print(f"[3d-batch] {index + 1}/{len(rows)} done")

    write_jsonl(Path(args.output), outputs)


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="HF model id or local checkpoint path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map. Use 'none' to disable.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, bfloat16, float16, or float32.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logs.")


def add_generation_args(parser: argparse.ArgumentParser, *, tokens_per_round: int) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--tokens-per-round", type=int, default=tokens_per_round)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 uses greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)


def add_choice_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--option-a", default="")
    parser.add_argument("--option-b", default="")
    parser.add_argument("--option-c", default="")
    parser.add_argument("--option-d", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniReason-Med 2D/3D inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_2d = subparsers.add_parser("2d", help="Run one 2D image example.")
    add_common_model_args(p_2d)
    add_generation_args(p_2d, tokens_per_round=256)
    add_choice_args(p_2d)
    p_2d.add_argument("--image", required=True)
    p_2d.add_argument("--question", required=True)
    p_2d.add_argument("--output", default=None)
    p_2d.set_defaults(func=run_2d)

    p_3d = subparsers.add_parser("3d", help="Run one 3D volume example.")
    add_common_model_args(p_3d)
    add_generation_args(p_3d, tokens_per_round=128)
    add_choice_args(p_3d)
    p_3d.add_argument("--volume", required=True, help=".npy/.npz volume path; .nii/.nii.gz needs nibabel.")
    p_3d.add_argument("--question", required=True)
    p_3d.add_argument("--target-size", type=parse_target_size, default=DEFAULT_3D_SIZE, help="D,H,W after resize.")
    p_3d.add_argument("--normalize", choices=["percentile", "minmax"], default="percentile")
    p_3d.add_argument("--output", default=None)
    p_3d.set_defaults(func=run_3d)

    p_2d_batch = subparsers.add_parser("2d-batch", help="Run a JSON/JSONL batch with question/image_path fields.")
    add_common_model_args(p_2d_batch)
    add_generation_args(p_2d_batch, tokens_per_round=256)
    p_2d_batch.add_argument("--input", required=True, help="JSON list or JSONL records.")
    p_2d_batch.add_argument("--image-root", default=None)
    p_2d_batch.add_argument("--output", required=True, help="Output JSON list.")
    p_2d_batch.add_argument("--max-samples", type=int, default=None)
    p_2d_batch.set_defaults(func=run_2d_batch)

    p_3d_batch = subparsers.add_parser("3d-batch", help="Run an M3D-style CSV batch.")
    add_common_model_args(p_3d_batch)
    add_generation_args(p_3d_batch, tokens_per_round=128)
    p_3d_batch.add_argument("--input-csv", required=True, help="CSV with Image Path, Question, Choice A-D columns.")
    p_3d_batch.add_argument("--data-root", default=None)
    p_3d_batch.add_argument("--output", required=True, help="Output JSONL path.")
    p_3d_batch.add_argument("--target-size", type=parse_target_size, default=DEFAULT_3D_SIZE, help="D,H,W after resize.")
    p_3d_batch.add_argument("--normalize", choices=["percentile", "minmax"], default="percentile")
    p_3d_batch.add_argument("--max-samples", type=int, default=None)
    p_3d_batch.set_defaults(func=run_3d_batch)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
