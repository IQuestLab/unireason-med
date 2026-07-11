# UniReason-Med Inference

This directory contains standalone Hugging Face inference code extracted from the original evaluation scripts. It supports:

- 2D image inference with `<|box_start|>[x1,y1,x2,y2]<|box_end|>` region crops.
- 3D volume inference by serializing a volume into 32 slices and using `<|box_start|>[x1,y1,z1,x2,y2,z2]<|box_end|>` cuboid crops.

## 2D Single Example

```bash
python code/inference/unireason_med_infer.py 2d \
  --model-path IQuestLab/UniReason-Med \
  --image /path/to/image.png \
  --question "What is the most likely diagnosis?" \
  --option-a "Pneumonia" \
  --option-b "Atelectasis" \
  --option-c "Pleural effusion" \
  --option-d "Normal"
```

## 3D Single Example

```bash
python code/inference/unireason_med_infer.py 3d \
  --model-path IQuestLab/UniReason-Med \
  --volume /path/to/volume.npy \
  --question "Which finding is present?" \
  --option-a "A" \
  --option-b "B" \
  --option-c "C" \
  --option-d "D"
```

3D inputs are resized to `32,256,256` by default and normalized with 1st/99th percentiles. `.npy` and `.npz` are supported directly; `.nii`/`.nii.gz` requires `nibabel`.

## Batch Formats

2D batch input can be a JSON list or JSONL with fields like:

```json
{"question": "...", "image_path": "relative/or/absolute.png", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "..."}
```

```bash
python code/inference/unireason_med_infer.py 2d-batch \
  --model-path IQuestLab/UniReason-Med \
  --input examples.jsonl \
  --image-root /path/to/images \
  --output predictions.json
```

3D batch input follows the M3D-style CSV columns used by the evaluation code: `Image Path`, `Question`, `Choice A`, `Choice B`, `Choice C`, `Choice D`.

```bash
python code/inference/unireason_med_infer.py 3d-batch \
  --model-path IQuestLab/UniReason-Med \
  --input-csv m3d_vqa.csv \
  --data-root /path/to/M3D-root \
  --output predictions.jsonl
```
