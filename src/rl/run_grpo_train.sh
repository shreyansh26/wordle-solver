#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# Required paths
TRAIN_JSONL=${TRAIN_JSONL:-"/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/train/moonshot_kimi_k2_data_train_v2_rl_train_use.jsonl"}
EVAL_JSONL=${EVAL_JSONL:-"/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/train/moonshot_kimi_k2_data_val_v2_rl_val_use.jsonl"}
VALID_WORDS=${VALID_WORDS:-"../data/valid_wordle_words.txt"}
MODEL_PATH=${MODEL_PATH:-"/mnt/ssd2/shreyansh/models/qwen3/exp_2025-08-08T00:53:20_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft/epoch_5/step_final"}
OUT_DIR=${OUT_DIR:-"/mnt/ssd2/shreyansh/models/qwen3/grpo_vllm_rl_v5"}

# vLLM server config
VLLM_HOST=${VLLM_HOST:-"0.0.0.0"}
VLLM_PORT=${VLLM_PORT:-9900}
VLLM_TIMEOUT=${VLLM_TIMEOUT:-300}
VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-""} # set if server model id differs

# Training hyperparams
BATCH=${BATCH:-1}
EVAL_BATCH=${EVAL_BATCH:-1}
ACC_STEPS=${ACC_STEPS:-8}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-6}
NUM_GEN=${NUM_GEN:-8}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-24576}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
MAX_CONCURRENT=${MAX_CONCURRENT:-512}
NUM_AHEAD=${NUM_AHEAD:-1}
ASYNC_TIMEOUT=${ASYNC_TIMEOUT:-600}
ASYNC_QUEUE=${ASYNC_QUEUE:-4}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-"cosine"}
WARMUP_STEPS=${WARMUP_STEPS:-50}

# WandB
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"wordle_grpo_rl"}
WANDB_ENTITY=${WANDB_ENTITY:-"shreyansh26"}
RUN_NAME=${RUN_NAME:-"grpo_vllm_rl_v5"}
LOG_COMPLETIONS=${LOG_COMPLETIONS:-1}
NUM_COMPLETIONS=${NUM_COMPLETIONS:-8}
LOG_UNIQUE=${LOG_UNIQUE:-0}

cd "$(dirname "$0")"/..

CMD=(python -m rl.train_grpo \
  --train_jsonl "$TRAIN_JSONL" \
  --valid_words "$VALID_WORDS" \
  --model_name_or_path "$MODEL_PATH" \
  --output_dir "$OUT_DIR" \
  --bf16 \
  --per_device_train_batch_size "$BATCH" \
  --per_device_eval_batch_size "$EVAL_BATCH" \
  --gradient_accumulation_steps "$ACC_STEPS" \
  --num_train_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --num_generations "$NUM_GEN" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_steps "$WARMUP_STEPS" \
  --temperature "$TEMP" --top_p "$TOP_P" --top_k "$TOP_K" --min_p "$MIN_P" \
  --use_think --max_turns 6 \
  --vllm_host "$VLLM_HOST" --vllm_port "$VLLM_PORT" --vllm_timeout "$VLLM_TIMEOUT" \
  --num_batches_ahead "$NUM_AHEAD" --async_generation_timeout "$ASYNC_TIMEOUT" --async_max_queue_size "$ASYNC_QUEUE" \
  --max_concurrent "$MAX_CONCURRENT")

if [[ -n "$EVAL_JSONL" ]]; then
  CMD+=(--eval_jsonl "$EVAL_JSONL")
fi

if [[ -n "$VLLM_MODEL_NAME" ]]; then
  CMD+=(--vllm_model_name "$VLLM_MODEL_NAME")
fi

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then CMD+=(--wandb_entity "$WANDB_ENTITY"); fi
  if [[ -n "$RUN_NAME" ]]; then CMD+=(--wandb_run_name "$RUN_NAME"); fi
  if [[ "$LOG_COMPLETIONS" == "1" ]]; then CMD+=(--log_completions --num_completions_to_print "$NUM_COMPLETIONS"); fi
  if [[ "$LOG_UNIQUE" ]] && [[ "$LOG_UNIQUE" == "1" ]]; then CMD+=(--wandb_log_unique_prompts); fi
fi

printf "[GRPO] Launching training...\n"
printf "Command: %q " "${CMD[@]}"; echo
"${CMD[@]}"



