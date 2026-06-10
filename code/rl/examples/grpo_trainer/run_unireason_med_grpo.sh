#!/usr/bin/env bash
set -euo pipefail
set -x
ulimit -n 65535

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

export VLLM_USE_V1="${VLLM_USE_V1:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/examples/urm_multiturn/config}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/unireason_med_rl}"

PROMPT_LENGTH="${PROMPT_LENGTH:-60000}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-4096}"

MODEL_PATH="${MODEL_PATH:-IQuestLab/UniReason-Med}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
TEST_DATA="${TEST_DATA:-$DATA_DIR/val.parquet}"

NNODES="${PET_NNODES:-${NNODES:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-6379}}"

RAY_PORT="${RAY_PORT:-${MASTER_PORT}}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
HEAD_ADDR="${HEAD_ADDR:-${MASTER_ADDR}}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-600}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)}}"
if [[ "${N_GPUS_PER_NODE}" -le 0 ]]; then
    echo "[WARN] No GPU detected, fallback N_GPUS_PER_NODE=1"
    N_GPUS_PER_NODE=1
fi

TOTAL_WORLD_SIZE=$((NNODES * N_GPUS_PER_NODE))
ROLLOUT_DP_HEADROOM_PER_NODE="${ROLLOUT_DP_HEADROOM_PER_NODE:-0}"
ROLLOUT_GPUS_PER_NODE_FOR_DP=$((N_GPUS_PER_NODE - ROLLOUT_DP_HEADROOM_PER_NODE))
if [[ "${ROLLOUT_GPUS_PER_NODE_FOR_DP}" -le 0 ]]; then
    echo "[WARN] ROLLOUT_DP_HEADROOM_PER_NODE=${ROLLOUT_DP_HEADROOM_PER_NODE} is too large for N_GPUS_PER_NODE=${N_GPUS_PER_NODE}, fallback ROLLOUT_GPUS_PER_NODE_FOR_DP=1"
    ROLLOUT_GPUS_PER_NODE_FOR_DP=1
fi

ROLLOUT_DP_TARGET_SIZE="${ROLLOUT_DP_SIZE:-$((NNODES * ROLLOUT_GPUS_PER_NODE_FOR_DP))}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_TARGET_SIZE}"
if [[ "${ROLLOUT_DP_SIZE}" -gt "${TOTAL_WORLD_SIZE}" ]]; then
    echo "[WARN] ROLLOUT_DP_SIZE=${ROLLOUT_DP_SIZE} exceeds total_world_size=${TOTAL_WORLD_SIZE}, clamp to ${TOTAL_WORLD_SIZE}"
    ROLLOUT_DP_SIZE="${TOTAL_WORLD_SIZE}"
fi
if [[ "${ROLLOUT_DP_SIZE}" -le 0 ]]; then
    echo "[WARN] Invalid ROLLOUT_DP_SIZE=${ROLLOUT_DP_SIZE}, fallback to 1"
    ROLLOUT_DP_SIZE=1
fi
while (( TOTAL_WORLD_SIZE % ROLLOUT_DP_SIZE != 0 )); do
    ROLLOUT_DP_SIZE=$((ROLLOUT_DP_SIZE - 1))
done
if [[ "${ROLLOUT_DP_SIZE}" -ne "${ROLLOUT_DP_TARGET_SIZE}" ]]; then
    echo "[WARN] Adjust rollout DP size from ${ROLLOUT_DP_TARGET_SIZE} to ${ROLLOUT_DP_SIZE} so total_world_size=${TOTAL_WORLD_SIZE} is divisible by infer_world_size"
fi

cd "$PROJECT_DIR"

echo "[INFO] PET_NNODES=${NNODES} PET_NODE_RANK=${NODE_RANK} PET_MASTER_ADDR=${MASTER_ADDR} PET_MASTER_PORT=${MASTER_PORT}"
echo "[INFO] Ray head=${HEAD_ADDR}:${RAY_PORT} dashboard=:${RAY_DASHBOARD_PORT} gpus_per_node=${N_GPUS_PER_NODE}"
echo "[INFO] Rollout DP size=${ROLLOUT_DP_SIZE} target=${ROLLOUT_DP_TARGET_SIZE} headroom_per_node=${ROLLOUT_DP_HEADROOM_PER_NODE} total_world_size=${TOTAL_WORLD_SIZE}"
echo "[INFO] Update weights bucket=${UPDATE_WEIGHTS_BUCKET_MB} MB"

ray stop --force >/dev/null 2>&1 || true

if [[ "${NODE_RANK}" == "0" ]]; then
    echo "[INFO] Starting Ray head..."
    ray start --head \
        --node-ip-address="${HEAD_ADDR}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --disable-usage-stats

    echo "[INFO] Waiting for workers to join..."
    HEAD_ADDR="${HEAD_ADDR}" RAY_PORT="${RAY_PORT}" PET_NNODES="${NNODES}" WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC}" python3 - <<'PYWAIT'
import os
import time

import ray

head = os.environ["HEAD_ADDR"]
port = os.environ["RAY_PORT"]
target = int(os.environ["PET_NNODES"])
timeout = int(os.environ["WAIT_TIMEOUT_SEC"])
addr = f"{head}:{port}"
deadline = time.time() + timeout

ray.init(address=addr, ignore_reinit_error=True, namespace="__probe__")
while True:
    alive = sum(1 for n in ray.nodes() if n.get("Alive"))
    print(f"[INFO] Ray alive nodes: {alive}/{target}", flush=True)
    if alive >= target:
        break
    if time.time() > deadline:
        raise TimeoutError(f"Timed out waiting for {target} nodes after {timeout}s")
    time.sleep(5)
ray.shutdown()
PYWAIT

    export RAY_ADDRESS=auto
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_vl_7b_unireason_med_grpo}"

    python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='unireason_med_2d3dmix_grpo_vllm' \
        algorithm.adv_estimator=grpo \
        data.train_files="$TRAIN_DATA" \
        data.val_files="$TEST_DATA" \
        data.image_key=images \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.max_prompt_length="${PROMPT_LENGTH}" \
        actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
        data.max_response_length="${RESPONSE_LENGTH}" \
        actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        +actor_rollout_ref.model.override_config.architectures='["Qwen2_5_VLForConditionalGeneration"]' \
        +actor_rollout_ref.model.override_config.text_config.architectures='["Qwen2_5_VLForConditionalGeneration"]' \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.agent.default_agent_loop=urm_single_assistant \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.architectures='["Qwen2_5_VLForConditionalGeneration"]' \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.min_pixels=3136 \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=12845056 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='verl_grpo_unireason_med' \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
        trainer.nnodes="${NNODES}" \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.total_epochs=3 \
        custom_reward_function.path="$PROJECT_DIR/examples/reward_function/unireason_med_mcq_reward.py" \
        custom_reward_function.name=compute_score \
        +custom_reward_function.reward_kwargs.answer_weight=0.8 \
        +custom_reward_function.reward_kwargs.format_weight=0.2 \
        "$@"
else
    echo "[INFO] Starting Ray worker, join ${HEAD_ADDR}:${RAY_PORT}..."
    ray start --address="${HEAD_ADDR}:${RAY_PORT}" --disable-usage-stats
    echo "[INFO] Worker joined Ray cluster, waiting..."
    tail -f /dev/null
fi
