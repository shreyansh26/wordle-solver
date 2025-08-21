import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

try:
    # optional convenience import if present in your repo
    from api_key import hf_token as DEFAULT_HF_TOKEN  # type: ignore
except Exception:
    DEFAULT_HF_TOKEN = None

from utils.state_dict_utils import to_hf


# Minimal copy of the args used to instantiate your Transformer
@dataclass
class ModelArgs:
    dim: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    ffn_dim_multiplier: float | None = None
    multiple_of: int | None = None
    norm_eps: float | None = None
    rope_theta: float | None = None
    use_scaled_rope: bool | None = None
    max_seq_len: int | None = None


def build_model_from_params(params_json: str, max_seq_len: Optional[int] = None) -> nn.Module:
    from model_llama import Transformer  # local project module

    with open(params_json, "r") as f:
        params = json.load(f)
    if max_seq_len is not None:
        params["max_seq_len"] = max_seq_len
    model_args = ModelArgs(**params)

    # meta init then empty to CPU to avoid needless allocations
    with torch.device("meta"):
        model = Transformer(model_args)
    model = model.to_empty(device="cpu")
    # Precompute freqs_cis like in training
    with torch.no_grad():
        model.freqs_cis = model._precompute_freqs_cis()
    return model


def apply_fsdp(model: nn.Module, param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16) -> None:
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    for layer_id, transformer_block in model.layers.named_children():
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)


def compile_model(model: nn.Module, backend: str = "inductor", fullgraph: bool = False) -> None:
    # Mirror training compile behavior: compile each transformer_block and re-register
    for layer_id, transformer_block in model.layers.named_children():
        compiled = torch.compile(transformer_block, backend=backend, fullgraph=fullgraph)
        model.layers.register_module(layer_id, compiled)


def _apply_ac_to_transformer_block(module: nn.Module):
    # Same wrapper used in training to ensure identical FQNs
    return ptd_checkpoint_wrapper(module, preserve_rng_state=False)


def apply_activation_checkpointing(model: nn.Module) -> None:
    for layer_id, transformer_block in model.layers.named_children():
        wrapped = _apply_ac_to_transformer_block(transformer_block)
        model.layers.register_module(layer_id, wrapped)


class AppState(Stateful):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # get_state_dict expects either a sequence of optimizers or None, not a single None
        optimizers = None if self.optimizer is None else [self.optimizer]
        model_sd, optim_sd = get_state_dict(self.model, optimizers)
        return {"model": model_sd, "optim": optim_sd}

    def load_state_dict(self, state_dict):
        # set_state_dict expects either a sequence of optimizers or None, not a single None
        optimizers = None if self.optimizer is None else [self.optimizer]
        set_state_dict(
            self.model,
            optimizers,
            model_state_dict=state_dict.get("model", {}),
            optim_state_dict=state_dict.get("optim", {}),
        )


def load_intermediate_checkpoint(ckpt_dir: str, model: nn.Module) -> None:
    """
    Load a distributed checkpoint saved with DCP into an FSDP2-sharded model.

    Expects that the FSDP plan (fully_shard over the same submodules) matches
    the plan used during training.
    """
    # Prepare a state dict placeholder matching the model's sharded layout
    # Preferred path: use Stateful wrapper so DCP computes the correct sharded plan
    try:
        state = {"app": AppState(model)}
        dcp_load(state, checkpoint_id=ckpt_dir)
        return
    except Exception as e_first:
        # Fallback path: manual placeholder (legacy saves)
        placeholder = get_model_state_dict(model)
        placeholder.pop("freqs_cis", None)
        load_errors: list[Exception] = []
        for candidate in (placeholder, {"model": placeholder}):
            try:
                dcp_load(candidate, checkpoint_id=ckpt_dir)
                loaded = candidate["model"] if isinstance(candidate, dict) and "model" in candidate else candidate
                set_model_state_dict(model, loaded, options=StateDictOptions(strict=False))
                return
            except Exception as e:
                load_errors.append(e)
        # none worked
        raise e_first
    # Recompute transient buffers excluded from checkpoint
    with torch.no_grad():
        if hasattr(model, "_precompute_freqs_cis"):
            model.freqs_cis = model._precompute_freqs_cis()


def init_dist_from_env():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    return local_rank, world_size, torch.device(f"cuda:{local_rank}")


def main():
    parser = argparse.ArgumentParser(description="Load FSDP2 DCP checkpoint")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to checkpoint dir (epoch_x/step_y)")
    parser.add_argument(
        "--params_json",
        type=str,
        default="./llama_3b_instruct/original/params.json",
        help="Path to model params.json (architecture only)",
    )
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override max_seq_len in params.json")
    parser.add_argument("--save_tokenizer_path", type=str, default=None, help="Optionally copy tokenizer to this path")
    parser.add_argument("--export_hf_dir", type=str, default=None, help="If set, consolidate to single HF checkpoint here")
    parser.add_argument("--hf_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="HF base model for config")
    parser.add_argument("--hf_token", type=str, default=None, help="HF auth token (optional)")
    parser.add_argument("--disable_compile", action="store_true", help="Disable torch.compile (default: enabled to match training)")
    parser.add_argument("--disable_ac", action="store_true", help="Disable activation checkpointing (default: enabled to match training)")
    args = parser.parse_args()

    local_rank, world_size, device = init_dist_from_env()
    if local_rank == 0:
        print(f"Loading DCP checkpoint from: {args.ckpt_dir}")

    # Build model and apply wrappers in training order: compile -> FSDP -> activation checkpointing
    model = build_model_from_params(args.params_json, args.max_seq_len)
    
    # Step 1: Compile (mirrors training line 477-480)
    if not args.disable_compile:
        torch._dynamo.config.capture_scalar_outputs = True
        compile_model(model, backend="inductor", fullgraph=False)
    
    # Step 2: Apply FSDP sharding (mirrors training line 482)
    apply_fsdp(model)
    
    # Step 3: Activation checkpointing after FSDP (mirrors training line 558-559)
    if not args.disable_ac:
        apply_activation_checkpointing(model)
    
    model.to(device)

    # Load DCP shards
    load_intermediate_checkpoint(args.ckpt_dir, model)

    # Load tokenizer saved at checkpoint dir
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    if args.save_tokenizer_path and local_rank == 0:
        os.makedirs(args.save_tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(args.save_tokenizer_path)

    # Optional: consolidate to single HF checkpoint
    if args.export_hf_dir is not None:
        # Important: call on all ranks to avoid hangs, but only rank 0 writes files
        cpu_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if local_rank == 0:
            os.makedirs(args.export_hf_dir, exist_ok=True)
            # Build model args again for to_hf adapter
            with open(args.params_json, "r") as f:
                params = json.load(f)
            if args.max_seq_len is not None:
                params["max_seq_len"] = args.max_seq_len
            model_args = ModelArgs(**params)

            # HF config for target architecture
            token = args.hf_token or DEFAULT_HF_TOKEN
            hf_cfg = AutoConfig.from_pretrained(args.hf_model_name, use_auth_token=token)

            # Convert and save
            hf_sd = to_hf(cpu_state, model_args)
            base = LlamaForCausalLM(hf_cfg)
            base.load_state_dict(hf_sd, strict=True)
            base.save_pretrained(args.export_hf_dir)
            tokenizer.save_pretrained(args.export_hf_dir)

    dist.barrier()
    if local_rank == 0:
        msg = "Checkpoint load complete."
        if args.export_hf_dir is not None:
            msg += f" HF export at: {args.export_hf_dir}"
        print(msg)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


