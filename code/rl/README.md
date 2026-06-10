# UniReason-Med RL

This directory contains the verl-based GRPO entry used for UniReason-Med. The released HF dataset stores RL records and 2D image bytes in separate Parquet splits, so materialize the training files once before launching verl.

```bash
cd code/rl
python examples/data_preprocess/unireason_med_2d3dmix.py --output-dir data/unireason_med_rl
MODEL_PATH=/path/to/sft/checkpoint bash examples/grpo_trainer/run_unireason_med_grpo.sh
```

`MODEL_PATH` should point to the SFT checkpoint used as the RL initialization. The script defaults to local materialized files:

```text
data/unireason_med_rl/train.parquet
data/unireason_med_rl/val.parquet
```

The materialized `images` values are relative local paths. 3D RL examples are text-only in the public release because the underlying M3D/Radiopaedia images require separate authorization.

Primary files:

```text
examples/data_preprocess/unireason_med_2d3dmix.py
examples/grpo_trainer/run_unireason_med_grpo.sh
examples/urm_multiturn/config/unireason_med_2d3dmix_grpo_vllm.yaml
examples/reward_function/unireason_med_mcq_reward.py
```

