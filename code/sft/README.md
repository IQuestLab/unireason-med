# UniReason-Med SFT

This directory contains the LLaMA-Factory-based supervised fine-tuning entry used for UniReason-Med. The released HF dataset stores SFT records and 2D image bytes in separate Parquet splits, so materialize the training files once before launching LLaMA-Factory.

```bash
cd code/sft
python scripts/materialize_unireason_med_sft.py --output-dir data/unireason_med_sft
llamafactory-cli train examples/train_full/unireason_med_sft.yaml
```

The materialized training file is `data/unireason_med_sft/train.parquet`. Its `images` values are relative paths under `data/unireason_med_sft/images/`. 3D records remain text-only in the release because the underlying M3D/Radiopaedia images require separate authorization.

Set `model_name_or_path` in `examples/train_full/unireason_med_sft.yaml` to the desired base or checkpoint path before training.
