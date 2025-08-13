# Wordle GRPO

End-to-end pipeline to train an LLM to solve Wordle, using:

- Supervised Fine-Tuning (SFT) on synthetic multi-turn traces with strict XML formatting - `<think> and </think> tags for reasoning and <guess> and </guess> tags for the final answer`
- Group Relative Policy Optimization (GRPO) with a verifiable multi-turn Wordle environment and a vLLM rollout server

The code is organized to:

- Generate Wordle play traces from a strong teacher (hosted on Together/Fireworks or a local model)
- Convert traces to SFT JSONL compatible with Qwen-style chat templates
- Train an SFT model (FSDP2, Flash-Attention 2, torch.compile) on Qwen3-4B (modifiable)
- Run GRPO over a strict Wordle environment backed by vLLM rollouts
- Evaluate/infer and compare performance across runs

---

## Repository layout

- `src/create_sft_data.py`: Generate Wordle multi-turn traces from a teacher model (Together/Fireworks/local). Produces `wordle_data_<WORD>.csv` per word.
- `src/analyze_data.py`: Summarize success/failure per word and produce a split list CSV.
- `src/prepare_training_data.py`: Convert trace CSVs into SFT/GRPO JSONL files with strict `<think>` and `<guess>` framing.
- `src/run_train_qwen3_fsdp.py`: SFT trainer for Qwen3-4B with FSDP + Flash-Attn2 + optional compile.
- `src/rl/wordle_env.py`: Strict multi-turn Wordle environment + rewards for GRPO.
- `src/rl/dataset.py`: Minimal GRPO dataset loader (JSONL with `{word, turns}` or `{prompt, answer}`).
- `src/rl/train_grpo.py`: GRPO entrypoint using the local `verifiers` package (rollouts via vLLM server).
- `src/rl/run_vllm_server.sh`: Start a vLLM server for rollouts.
- `src/rl/run_grpo_train.sh`: Shell wrapper to launch GRPO training with environment overrides.
- `src/run_inference.py`: Batch inference/eval against an LLM via OpenAI-compatible endpoints (local vLLM or remote).
- `src/utils/*`: Utilities (performance comparison, duplicate checks, etc.).
- `data/valid_wordle_words.txt`, `data/word_list.txt`: Word lists used by the environment and data generation.

---

## Requirements

- Python 3.11 or 3.12
- CUDA 12.x GPU recommended (training Qwen3-4B with FSDP + Flash-Attn2)

Core Python packages:

```
torch==2.6.0+cu124
transformers==4.53.2
trl==0.20.0
accelerate==1.8.1
datasets==4.0.0
flash_attn==2.5.8
wandb==0.21.0
python-dotenv==1.1.1
verifiers==0.1.2
vllm==0.10.0
```

Example env setup (CUDA 12.4):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Install core training deps (adjust torch/cu to your system)
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
pip install transformers==4.53.2 trl==0.20.0 accelerate==1.8.1 datasets==4.0.0 flash_attn==2.5.8 wandb==0.21.0 python-dotenv==1.1.1

# Install vLLM separately (match your CUDA/toolchain)
pip install vllm

# Install local verifiers package (provides env/trainer/server wrappers)
pip install -e verifiers
```

---

## Data generation pipeline (teacher rollouts → CSV → JSONL)

1) Generate teacher traces

`src/create_sft_data.py` queries a teacher model (Together.ai or Fireworks; or a local OpenAI-compatible endpoint) to produce per-word multi-turn transcripts with strict XML format:

```xml
<think>...</think>
<guess>apple</guess>
```

Configure provider and model at the top of `src/create_sft_data.py`:

- `response_provider`: one of `"together"`, `"fireworks"`, or `"local"`
- `model_name`: provider model id (e.g., `moonshotai/Kimi-K2-Instruct` or a local vLLM model id)

Auth and paths:

- Put your keys in environment variables (recommended) or a `.env`: `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`.
- Output directory: `../data/sft/<model_name_with_slashes_replaced>/wordle_data_<WORD>.csv`

Run:

```bash
cd src
python create_sft_data.py
```

2) Summarize success/failure per word

```bash
cd src
python analyze_data.py
# writes ../data/sft/train/moonshot_kimi_k2_summary_v2.csv
```

3) Build SFT/GRPO JSONL files

`src/prepare_training_data.py` reads the per-word CSVs and summary, enforces formatting/length constraints, and writes JSONL splits:

- SFT train: `../data/sft/train/moonshot_kimi_k2_data_train_v2_sft_train.jsonl`
- SFT val:   `../data/sft/train/moonshot_kimi_k2_data_val_v2_sft_val.jsonl`
- RL train:  `../data/sft/train/moonshot_kimi_k2_data_train_v2_rl_train.jsonl`
- RL val:    `../data/sft/train/moonshot_kimi_k2_data_val_v2_rl_val.jsonl`

Run:

```bash
cd src
python prepare_training_data.py
```

JSONL schema accepted by GRPO (`src/rl/dataset.py`):

- Option A: `{ "prompt": <chat_messages>, "answer": "apple" }`
- Option B: `{ "word": "apple", "turns": [{"guess": "arise", "feedback": "..."}, ...] }`

The SFT JSONL uses fields `{instruction, output}` where `instruction` is the chat template up to the assistant generation and `output` is the assistant content with `<think>` and `<guess>` plus the tokenizer EOS.

---

## SFT training (Qwen3-4B, FSDP + Flash-Attn2)

Entrypoint: `src/run_train_qwen3_fsdp.py`

Key points:

- Model: defaults to `Qwen/Qwen3-4B` (bf16, Flash-Attn2)
- Data: set `train_ds` and `val_ds` inside the script to your JSONL paths (by default points to the files produced above)
- Uses an HF token via `api_key.hf_token` (or set `HUGGINGFACE_HUB_TOKEN` and modify code accordingly)
- FSDP shards decoder layers + model; gradient checkpointing enabled; optional `--dcp-api` for CPU checkpoint offload save

Example (2 GPUs):

```bash
cd src
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 run_train_qwen3_fsdp.py --dcp-api
```

Artifacts are saved under `output_dir` composed from timestamp + notes (see script). Each epoch saves a checkpoint directory containing model and tokenizer.

---

## GRPO training (with vLLM rollouts)

GRPO consumes the SFT checkpoint as the initial policy and performs multi-turn rollouts against a strict Wordle environment with rewards for:

- Correct answer, early finish bonus, strict XML format adherence, valid constraints, 5-letter guess length

1) Start vLLM server

```bash
# Edit MODEL (HF hub id or local checkpoint path) and GPU parallelism
cd src/rl
MODEL=/path/to/your/sft_checkpoint \
HOST=0.0.0.0 PORT=9900 TP=1 DP=1 MAX_MODEL_LEN=32768 \
bash run_vllm_server.sh
```

2) Launch GRPO training

Use the shell wrapper and override inputs as needed (defaults point inside this repo):

```bash
cd src/rl
TRAIN_JSONL=/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/train/moonshot_kimi_k2_data_train_v2_rl_train.jsonl \
EVAL_JSONL=/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/train/moonshot_kimi_k2_data_val_v2_rl_val.jsonl \
VALID_WORDS=../data/valid_wordle_words.txt \
MODEL_PATH=/path/to/your/sft_checkpoint \
OUT_DIR=/mnt/ssd2/shreyansh/models/qwen3/grpo_run \
VLLM_HOST=0.0.0.0 VLLM_PORT=9900 \
ACC_STEPS=8 EPOCHS=1 LR=3e-6 NUM_GEN=8 MAX_SEQ_LEN=4096 \
WARMUP_STEPS=60 LR_SCHEDULER_TYPE=constant_with_warmup \
USE_WANDB=1 WANDB_PROJECT=wordle_grpo_rl RUN_NAME=grpo_vllm_test \
bash run_grpo_train.sh
```

Alternatively, call the Python entrypoint directly:

```bash
cd src
python -m rl.train_grpo \
  --train_jsonl /path/to/rl_train.jsonl \
  --eval_jsonl /path/to/rl_val.jsonl \
  --valid_words ../data/valid_wordle_words.txt \
  --model_name_or_path /path/to/your/sft_checkpoint \
  --output_dir /path/to/out \
  --bf16 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 --num_train_epochs 1 \
  --learning_rate 3e-6 --num_generations 8 \
  --max_seq_len 4096 --max_prompt_length 24576 \
  --lr_scheduler_type constant_with_warmup --warmup_steps 60 \
  --use_think --max_turns 6 \
  --vllm_host 0.0.0.0 --vllm_port 9900 --vllm_timeout 300 \
  --num_batches_ahead 1 --async_generation_timeout 600 --async_max_queue_size 4 \
  --max_concurrent 512 \
  --wandb --wandb_project wordle_grpo --wandb_run_name grpo_run
```

Note: Ensure the vLLM server is reachable and serving the same (or an aliased) model id expected by clients. You can override the rollout model name via `--vllm_model_name` if needed.

---

## Inference/evaluation

`src/run_inference.py` can probe a set of words using a local vLLM server (OpenAI-compatible) and write a summary CSV per model.

Configure at the top of the file:

- `MODEL_NAME`: local checkpoint path or hub id
- `base_url`: points to your server (defaults to `http://localhost:9203/v1` in this file; adjust as needed)

Run:

```bash
cd src
python run_inference.py
```

You can compare summary CSVs with `src/utils/compare_performance.py`

---
