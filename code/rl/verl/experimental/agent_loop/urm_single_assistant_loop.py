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
import logging
import os
from io import BytesIO
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.urm_bbox_utils import (
    crop_2d_image,
    extract_last_2d_bbox,
    generate_random_bbox,
    resize_pil_to_processor,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("urm_single_assistant")
class URMSingleAssistantLoop(AgentLoopBase):
    """
    URM-style single-assistant multi-round loop:
    - Generate until EOS or <|box_end|>
    - If ending at <|box_end|>, crop image and continue generation
    - Keep output semantics as one continuous assistant reply
    """

    ASSISTANT_DELIMITER = "<|im_end|>\n<|im_start|>assistant\n"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.max_rounds = int(self.config.actor_rollout_ref.rollout.get("urm_max_rounds", 10))
        self.debug_log_file = os.getenv("URM_SINGLE_ASSISTANT_LOG_FILE", "/tmp/urm_single_assistant_loop.log")

        self.enable_contrast_reward = bool(
            self.config.actor_rollout_ref.rollout.get("urm_enable_contrast_reward", False)
        )
        self.contrast_logprobs_k = int(
            self.config.actor_rollout_ref.rollout.get("urm_contrast_logprobs_k", 20)
        )

        self.box_end_token_id: Optional[int] = None
        try:
            token_id = self.tokenizer.convert_tokens_to_ids("<|box_end|>")
            if token_id is not None and token_id >= 0:
                self.box_end_token_id = int(token_id)
        except Exception:
            self.box_end_token_id = None

        self.eos_token_id = self.tokenizer.eos_token_id

        # Precompute answer token IDs for ABCD MCQ choices.
        self._answer_token_ids: dict[str, int] = {}
        for ch in ("A", "B", "C", "D"):
            for variant in (ch, f" {ch}"):
                try:
                    ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    if len(ids) == 1:
                        self._answer_token_ids[ch] = ids[0]
                        break
                except Exception:
                    pass

        # Precompute </think> token ID for answer position detection.
        self._think_end_token_id: Optional[int] = None
        try:
            tid = self.tokenizer.convert_tokens_to_ids("</think>")
            if tid is not None and tid >= 0:
                self._think_end_token_id = int(tid)
        except Exception:
            pass

    def _append_debug_log(self, message: str) -> None:
        try:
            log_dir = os.path.dirname(self.debug_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.debug_log_file, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        except Exception:
            logger.exception("Failed to write URM debug log to %s", self.debug_log_file)

    def _compact_assistant_delimiter(self, text: str) -> str:
        """Keep only the first assistant delimiter to enforce continuous single-assistant generation."""
        first_idx = text.find(self.ASSISTANT_DELIMITER)
        if first_idx < 0:
            return text
        before_first = text[: first_idx + len(self.ASSISTANT_DELIMITER)]
        after_first = text[first_idx + len(self.ASSISTANT_DELIMITER) :]
        if self.ASSISTANT_DELIMITER in after_first:
            after_first = after_first.replace(self.ASSISTANT_DELIMITER, "")
        return before_first + after_first

    def _collect_images_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        image_inputs: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "image":
                    continue
                if "image" in item:
                    image_inputs.append(item["image"])
                elif "image_url" in item:
                    image_inputs.append(item["image_url"])
        return image_inputs

    def _to_pil_image(self, image: Any) -> Optional[Image.Image]:
        """Best-effort conversion for image-like inputs used by raw_prompt."""
        try:
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            if isinstance(image, (str, os.PathLike)):
                return Image.open(os.fspath(image)).convert("RGB")
            if isinstance(image, dict):
                if "image" in image:
                    return self._to_pil_image(image["image"])
                if "bytes" in image:
                    return Image.open(BytesIO(image["bytes"])).convert("RGB")
                if "image_url" in image:
                    image_url = image["image_url"]
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    if isinstance(image_url, (str, os.PathLike)) and os.path.exists(os.fspath(image_url)):
                        return Image.open(os.fspath(image_url)).convert("RGB")
        except Exception:
            return None
        return None

    async def _build_prompt_ids(self, messages: list[dict[str, Any]]) -> tuple[list[int], list[Any]]:
        """Build prompt ids from full messages and return images in message order."""
        if self.processor is None:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            return prompt_ids, []

        # Keep image parsing behavior consistent with standard agent loops.
        # qwen_vl_utils will normalize raw_prompt image entries (e.g. file paths) to PIL images.
        image_inputs: list[Any] = []
        try:
            multi_modal_data = await self.process_vision_info(messages)
            image_inputs = list(multi_modal_data.get("images", []) or [])
        except Exception:
            image_inputs = []
        if not image_inputs:
            image_inputs = self._collect_images_from_messages(messages)
        raw_prompt = await self.loop.run_in_executor(
            None,
            lambda: self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        raw_prompt = self._compact_assistant_delimiter(raw_prompt)
        # print("\nRaw Prompt:")
        # print(raw_prompt)
        # print("\nimage_inputs:")
        # print(image_inputs)
        
        # import pdb; pdb.set_trace()
        model_inputs = self.processor(
            text=[raw_prompt],
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            do_sample_frames=False,
        )
        prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        return prompt_ids, image_inputs

    @staticmethod
    def _extract_observation_ids(
        current_prompt_ids: list[int], generated_ids: list[int], next_prompt_ids: list[int]
    ) -> list[int]:
        """
        Extract non-generated observation token ids injected by the environment between rounds.
        Expected case:
            next_prompt = current_prompt + generated + observation
        """
        expected_prefix = current_prompt_ids + generated_ids
        expected_len = len(expected_prefix)
        if len(next_prompt_ids) >= expected_len and next_prompt_ids[:expected_len] == expected_prefix:
            return next_prompt_ids[expected_len:]

        # Fallback: use longest common prefix to avoid crash on template/tokenization edge cases.
        lcp = 0
        max_len = min(len(next_prompt_ids), expected_len)
        while lcp < max_len and next_prompt_ids[lcp] == expected_prefix[lcp]:
            lcp += 1
        return next_prompt_ids[lcp:]

    def _should_continue_with_crop(self, generated_ids: list[int]) -> bool:
        return self.box_end_token_id is not None and len(generated_ids) > 0 and generated_ids[-1] == self.box_end_token_id

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages: list[dict[str, Any]] = list(kwargs["raw_prompt"])

        prompt_ids, image_inputs = await self._build_prompt_ids(messages)
        initial_prompt_ids = list(prompt_ids)
        current_prompt_ids = list(prompt_ids)
        current_images = list(image_inputs)

        self._append_debug_log(
            (
                f"[pid={os.getpid()}] run_start "
                f"log_file={self.debug_log_file} "
                f"box_end_token_id={self.box_end_token_id} "
                f"prompt_len={len(initial_prompt_ids)} "
                f"num_images={len(current_images)} "
                f"first_image_type={type(current_images[0]).__name__ if current_images else 'none'}"
            )
        )

        base_image_aligned = None
        if self.processor is not None and current_images:
            base_image = self._to_pil_image(current_images[0])
            if base_image is not None:
                base_image_aligned = resize_pil_to_processor(base_image, self.processor)

        response_ids: list[int] = []
        response_mask: list[int] = []
        assistant_turns = 0
        metrics: dict[str, float] = {"tool_calls": 0.0}

        logprob_requested = bool(sampling_params.get("logprobs", False))
        response_logprobs: Optional[list[float]] = [] if logprob_requested else None

        # State variables for contrastive grounding reward tracking.
        _crop_happened: bool = False
        _num_crops: int = 0
        _last_pred_bbox: Optional[list[float]] = None
        _messages_before_crop_turn: Optional[list[dict]] = None
        _round1_response_text: Optional[str] = None
        _round2_ids: Optional[list[int]] = None

        # for _ in range(self.max_rounds):
        for round_index in range(self.max_rounds):
            if len(response_mask) >= self.response_length:
                break

            round_sampling_params = dict(sampling_params)
            round_sampling_params["max_tokens"] = self.response_length - len(response_mask)

            # Stop at either EOS or box_end to allow crop-and-continue.
            stop_token_ids = []
            if self.eos_token_id is not None:
                stop_token_ids.append(int(self.eos_token_id))
            if self.box_end_token_id is not None:
                stop_token_ids.append(int(self.box_end_token_id))
            if stop_token_ids:
                round_sampling_params["stop_token_ids"] = sorted(set(stop_token_ids))

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=current_prompt_ids,
                    sampling_params=round_sampling_params,
                    image_data=current_images if current_images else None,
                    video_data=None,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            generated_ids = output.token_ids
            if not generated_ids:
                self._append_debug_log(f"[pid={os.getpid()}] break round={round_index}: empty generation")
                break

            assistant_turns += 1
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))

            if response_logprobs is not None:
                if output.log_probs is not None and len(output.log_probs) == len(generated_ids):
                    response_logprobs.extend(output.log_probs)
                else:
                    response_logprobs.extend([0.0] * len(generated_ids))

            if len(response_mask) >= self.response_length:
                self._append_debug_log(
                    f"[pid={os.getpid()}] break round={round_index}: reach response_length={self.response_length}"
                )
                break

            # URM round continuation: crop and append image when terminated by box_end.
            if not self._should_continue_with_crop(generated_ids):
                terminal_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=False),
                )
                self._append_debug_log(
                    (
                        f"[pid={os.getpid()}] break round={round_index}: "
                        f"no_box_end last_token={generated_ids[-1]} "
                        f"response_text={terminal_text}"
                    )
                )
                # Record round2 ids if a crop happened in a prior round.
                if _crop_happened:
                    _round2_ids = list(generated_ids)
                break

            if base_image_aligned is None:
                self._append_debug_log(
                    f"[pid={os.getpid()}] break round={round_index}: base_image_aligned is None"
                )
                break

            response_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=False),
            )
            bbox_2d = extract_last_2d_bbox(response_text)
            if bbox_2d is None:
                self._append_debug_log(
                    f"[pid={os.getpid()}] final round={round_index}, response_text={response_text}"
                )
                break

            with simple_timer("tool_calls", metrics):
                try:
                    cropped_image = crop_2d_image(base_image_aligned, bbox_2d)
                except Exception as e:
                    self._append_debug_log(
                        f"[pid={os.getpid()}] break round={round_index}: crop_2d_image failed: {repr(e)}"
                    )
                    break

                # Snapshot state before appending the crop assistant turn (for contrast reward).
                _messages_before_crop_turn = [dict(m) for m in messages]
                _last_pred_bbox = bbox_2d
                _round1_response_text = response_text

                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": response_text},
                            {"type": "image", "image": cropped_image},
                        ],
                    }
                )
                self._append_debug_log(
                    f"[pid={os.getpid()}] round_index={round_index}, messages={messages}"
                )

                next_prompt_ids, next_images = await self._build_prompt_ids(messages)
                observation_ids = self._extract_observation_ids(current_prompt_ids, generated_ids, next_prompt_ids)

                if observation_ids:
                    remaining = self.response_length - len(response_mask)
                    if len(observation_ids) > remaining:
                        # Do not append partial observation tokens; it can desync image placeholders
                        # with multi-modal inputs and break rope position computation.
                        self._append_debug_log(
                            (
                                f"[pid={os.getpid()}] break round={round_index}: "
                                f"observation_overflow obs_len={len(observation_ids)} remaining={remaining}"
                            )
                        )
                        break

                    response_ids.extend(observation_ids)
                    response_mask.extend([0] * len(observation_ids))
                    if response_logprobs is not None:
                        response_logprobs.extend([0.0] * len(observation_ids))

                # Advance prompt/image state only when the observation is fully appended.
                current_prompt_ids = next_prompt_ids
                current_images = list(next_images)

                _crop_happened = True
                _num_crops += 1

            if len(response_mask) >= self.response_length:
                self._append_debug_log(
                    f"[pid={os.getpid()}] break round={round_index}: reach response_length={self.response_length}"
                )
                # Record round2 ids if a crop happened and next generation would be round2.
                if _crop_happened and _round2_ids is None:
                    # round2 generation hasn't happened yet because we hit response_length during obs injection
                    pass
                break

        final_multi_modal_data: dict[str, Any] = {}
        if current_images:
            final_multi_modal_data["images"] = current_images

        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        if response_logprobs is not None:
            response_logprobs = response_logprobs[: self.response_length]

        # Compute contrastive grounding reward when enabled and a crop happened.
        rand_crop_pred_choice: Optional[str] = None
        if self.enable_contrast_reward and _crop_happened and _round2_ids:
            rand_crop_pred_choice = await self._compute_contrastive_generation(
                messages_before_crop=_messages_before_crop_turn,
                round1_response_text=_round1_response_text,
                pred_bbox=_last_pred_bbox,
                base_image_aligned=base_image_aligned,
                sampling_params=sampling_params,
            )

        # Compute crop area ratio: (bbox_w * bbox_h) / (img_w * img_h).
        _crop_area_ratio = 0.0
        if _crop_happened and _last_pred_bbox is not None and base_image_aligned is not None:
            x1, y1, x2, y2 = _last_pred_bbox
            img_w, img_h = base_image_aligned.size
            img_area = img_w * img_h
            if img_area > 0:
                crop_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                _crop_area_ratio = min(1.0, crop_area / img_area)

        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=final_multi_modal_data,
            num_turns=1 + assistant_turns,
            metrics=metrics,
            extra_fields={
                "urm_crop_happened": _crop_happened,
                "urm_num_crops": _num_crops,
                "urm_rand_crop_pred_choice": rand_crop_pred_choice or "",
                "urm_crop_area_ratio": _crop_area_ratio,
            },
        )

    def _extract_answer_from_text(self, text: str) -> str:
        """Extract the MCQ answer choice (A/B/C/D) from generated text."""
        import re as _re
        if not text:
            return ""
        plain = _re.sub(r"<think>.*?</think>", " ", text, flags=_re.IGNORECASE | _re.DOTALL)
        m = _re.search(r"\b([A-D])\b", plain, flags=_re.IGNORECASE)
        if not m:
            return ""
        return m.group(1).upper()

    async def _compute_contrastive_generation(
        self,
        *,
        messages_before_crop: list[dict],
        round1_response_text: str,
        pred_bbox: list[float],
        base_image_aligned: Optional[Image.Image],
        sampling_params: dict,
    ) -> Optional[str]:
        """Generate with a random crop and return the predicted answer choice (A/B/C/D)."""
        if base_image_aligned is None:
            return None

        # 1. Generate a random bbox and crop.
        try:
            rand_bbox = generate_random_bbox(base_image_aligned, pred_bbox)
            rand_crop = crop_2d_image(base_image_aligned, rand_bbox)
        except Exception:
            logger.exception("Failed to generate random crop for contrastive generation")
            return None

        # 2. Build messages with rand_crop and get prompt ids.
        rand_messages = list(messages_before_crop) + [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": round1_response_text},
                    {"type": "image", "image": rand_crop},
                ],
            }
        ]
        try:
            rand_prompt_ids, rand_images = await self._build_prompt_ids(rand_messages)
        except Exception:
            logger.exception("Failed to build prompt ids for random crop contrastive generation")
            return None

        # 3. Generate round2 response — stop only at EOS, no logprobs needed.
        rand_sampling_params = dict(sampling_params)
        stop_token_ids = []
        if self.eos_token_id is not None:
            stop_token_ids.append(int(self.eos_token_id))
        if stop_token_ids:
            rand_sampling_params["stop_token_ids"] = stop_token_ids
        rand_sampling_params["max_tokens"] = self.response_length
        rand_sampling_params.pop("logprobs", None)

        try:
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=rand_prompt_ids,
                sampling_params=rand_sampling_params,
                image_data=rand_images if rand_images else None,
                video_data=None,
            )
        except Exception:
            logger.exception("Failed to generate for random crop contrastive generation")
            return None

        generated_ids = output.token_ids
        if not generated_ids:
            return None

        # 4. Decode and extract answer choice.
        try:
            generated_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=False),
            )
        except Exception:
            return None

        return self._extract_answer_from_text(generated_text)
