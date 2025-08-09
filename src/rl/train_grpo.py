from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import verifiers as vf
from verifiers.trainers.grpo_config import GRPOConfig
from verifiers.trainers.grpo_trainer import GRPOTrainer

from rl.dataset import load_jsonl_as_dataset
from rl.wordle_env import load_environment as load_wordle_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default=None)
    p.add_argument("--valid_words", type=str, required=True)
    p.add_argument("--max_train_rows", type=int, default=-1)
    p.add_argument("--max_eval_rows", type=int, default=-1)
    # Model
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--bf16", action="store_true")
    # Training
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    # Generation / GRPO
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument(
        "--max_prompt_length",
        type=int,
        default=None,
        help="Optional cap on prompt tokens; if unset, derived from max_seq_len.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max generation tokens sent to vLLM. If unset, derived from max_seq_len and max_prompt_length.",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--max_concurrent", type=int, default=512)
    p.add_argument("--num_batches_ahead", type=int, default=1)
    # Multi-turn env
    p.add_argument("--use_think", action="store_true")
    p.add_argument("--max_turns", type=int, default=6)
    # vLLM server rollouts
    p.add_argument("--vllm_host", type=str, default="0.0.0.0")
    p.add_argument("--vllm_port", type=int, default=8000)
    p.add_argument("--vllm_timeout", type=float, default=300.0)
    p.add_argument("--async_generation_timeout", type=float, default=600.0)
    p.add_argument("--async_max_queue_size", type=int, default=None)
    p.add_argument("--vllm_model_name", type=str, default=None, help="Override model name sent to vLLM server for rollouts")
    # Weights & Biases
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="wordle_grpo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--log_completions", action="store_true", help="Log sample prompts/completions to wandb")
    p.add_argument("--num_completions_to_print", type=int, default=8)
    p.add_argument("--wandb_log_unique_prompts", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    # Load datasets from JSONL
    train_ds = load_jsonl_as_dataset(args.train_jsonl, use_think=args.use_think, limit=args.max_train_rows)
    eval_ds = (
        load_jsonl_as_dataset(args.eval_jsonl, use_think=args.use_think, limit=args.max_eval_rows)
        if args.eval_jsonl
        else None
    )

    # Build environment
    env = load_wordle_env(
        valid_words_path=args.valid_words,
        dataset=train_ds,
        eval_dataset=eval_ds,
        use_think=args.use_think,
        max_turns=args.max_turns,
    )

    # Model + tokenizer
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)

    # If the vLLM server expects a different model identifier, override the name used in rollout requests
    if args.vllm_model_name:
        try:
            model.config._name_or_path = args.vllm_model_name  # type: ignore
        except Exception:
            pass

    # WANDB setup
    report_to = ["wandb"] if args.wandb else ["none"]
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    # Derive prompt/completion lengths to satisfy vLLM max context
    # Ensure: max_prompt_length + max_tokens <= max_seq_len
    # if args.max_prompt_length is not None and args.max_tokens is not None:
    #     eff_max_prompt_len = min(args.max_prompt_length, args.max_seq_len - 1)
    #     eff_max_tokens = min(args.max_tokens, max(1, args.max_seq_len - eff_max_prompt_len))
    # elif args.max_prompt_length is not None:
    #     eff_max_prompt_len = min(args.max_prompt_length, args.max_seq_len - 1)
    #     eff_max_tokens = max(1, args.max_seq_len - eff_max_prompt_len)
    # elif args.max_tokens is not None:
    #     eff_max_tokens = min(args.max_tokens, args.max_seq_len - 1)
    #     eff_max_prompt_len = max(1, args.max_seq_len - eff_max_tokens)
    # else:
    #     # Default split: 1/3 prompt, 2/3 completion (safe for long completions)
    #     eff_max_prompt_len = max(256, args.max_seq_len // 3)
    #     eff_max_tokens = max(1, args.max_seq_len - eff_max_prompt_len)

    # GRPO config
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=(args.wandb_run_name or f"grpo-{Path(args.model_name_or_path).name}"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        do_eval=eval_ds is not None,
        seed=args.seed,

        # Generation params
        num_generations=args.num_generations,
        # max_seq_len=args.max_seq_len,
        # max_prompt_length=eff_max_prompt_len,
        # max_tokens=eff_max_tokens,
        max_seq_len=args.max_seq_len,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_concurrent=args.max_concurrent,
        num_batches_ahead=args.num_batches_ahead,

        # Mask environment messages in loss
        mask_env_responses=True,
        mask_truncated_completions=True,
        zero_truncated_completions=False,

        report_to=report_to,
        remove_unused_columns=False,
        fp16=not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=True,

        # vLLM rollout configuration
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        vllm_server_timeout=args.vllm_timeout,
        async_generation_timeout=args.async_generation_timeout,
        async_max_queue_size=args.async_max_queue_size,

        # Textual logging / tables
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        wandb_log_unique_prompts=args.wandb_log_unique_prompts,
    )

    trainer = GRPOTrainer(
        model=model,
        env=env,
        args=grpo_args,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()