from api_key import hf_token
from utils.training_utils import (
    SupervisedDataset,
    DataCollatorForSupervisedDataset,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from tqdm import tqdm
from datetime import datetime
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter as StorageWriter
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed._tensor import DTensor, distribute_tensor
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaAttention, LlamaMLP
from model_llama import Transformer
from utils.state_dict_utils import to_hf
from dotenv import load_dotenv
import torch.distributed as dist
import wandb
import torch
import torch.nn as nn
import transformers
import os
import math
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import json
from contextlib import contextmanager, ExitStack
import argparse
from typing import Iterable
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except ImportError:
    # Fallback to pytz for older Pythons
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

load_dotenv()

@dataclass
class ModelArgs:
    dim: int = None
    n_layers: int = None
    n_heads: int = None
    n_kv_heads: int = None
    vocab_size: int = None
    ffn_dim_multiplier: float = None
    multiple_of: int = None
    norm_eps: float = None
    rope_theta: float = None
    use_scaled_rope: bool = None
    max_seq_len: int = None

def current_timestamp_ist() -> str:
    """
    Returns the current time in IST (Asia/Kolkata) as an ISO-like string:
      YYYY-MM-DDTHH:MM:SS
    e.g. "2025-07-10T14:30:05"
    """
    now = datetime.now(tz=IST)
    return now.strftime("%Y-%m-%dT%H:%M:%S")

def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy loss with shift (predict t+1 from tokens up to t)."""
    # pred: [batch, seq_len, vocab_size], labels: [batch, seq_len]
    if pred.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected pred shape [batch, seq_len, vocab] and labels [batch, seq_len]")

    # shift for next-token prediction
    pred = pred[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()

    # Use SUM reduction and divide by valid tokens for true per-token average
    # print("Pred: ", pred, "Local rank: ", local_rank)
    # print("Labels: ", labels, "Local rank: ", local_rank)

    loss_sum = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    # Count valid tokens (non-padding)
    valid_tokens = (labels.reshape(-1) != -100).sum().float()
    
    # Return per-token loss, handling empty segments gracefully
    if valid_tokens > 0:
        return loss_sum / valid_tokens
    else:
        # Return small loss for completely empty segments (instead of NaN)
        return torch.tensor(0.01, device=pred.device, dtype=pred.dtype, requires_grad=True)

def load_model(model_path, model_args, use_flash_attn_api: bool = False, use_flash_attn_sdpa: bool = False, tp_enabled: bool = False):
    with torch.device("meta"):
        model = Transformer(model_args, use_flash_attn_api=use_flash_attn_api, use_flash_attn_sdpa=use_flash_attn_sdpa)
    
    model = model.to_empty(device="cpu")
    state_dict = torch.load(f"{model_path}/consolidated.00.pth", weights_only=True, mmap=True)
    
    if tp_enabled:
        model.load_state_dict(state_dict)
    else: 
        model.load_state_dict(state_dict, assign=True)

    # Load freqs_cis separately
    with torch.no_grad():
        model.freqs_cis = model._precompute_freqs_cis()
    return model

def setup_model(model_name, max_length, use_flash_attn_api: bool = False, use_flash_attn_sdpa: bool = False, tp_enabled: bool = False):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        token=hf_token,
    )
    config.use_cache = False

    model_name_original = "llama_3b_instruct"
    model_path = f"./{model_name_original}/original"
    model_config = f"{model_path}/params.json"
    with open(model_config, "r") as f:
        params = json.load(f)

    params['max_seq_len'] = 131072
    model_args = ModelArgs(**params)

    model = load_model(model_path, model_args, use_flash_attn_api=use_flash_attn_api, use_flash_attn_sdpa=use_flash_attn_sdpa, tp_enabled=tp_enabled)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer, config, model_args

def compile_model(model, backend="inductor", fullgraph=False):
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, backend=backend, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)

    if local_rank == 0:
        print("Compiling each TransformerBlock with torch.compile")

def apply_fsdp(model, dp_mesh: DeviceMesh | None = None, tp_enabled: bool = False, cp_enabled: bool = False):
    # For CP, avoid resharding after forward to prevent issues with sequence sharding
    reshard_after_forward = not (tp_enabled or cp_enabled)
    
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        "reshard_after_forward": reshard_after_forward,
    }
    if dp_mesh is not None:
        fsdp_kwargs["mesh"] = dp_mesh

    if hasattr(model, 'tok_embeddings') and model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, **fsdp_kwargs)
    
    # Apply to transformer layers
    for layer_id, transformer_block in model.layers.named_children():
        fully_shard(transformer_block, **fsdp_kwargs)
    
    # Apply to norm and output with optimized resharding
    if hasattr(model, 'norm') and hasattr(model, 'output'):
        if model.norm is not None and model.output is not None:
            norm_output_kwargs = fsdp_kwargs.copy()
            norm_output_kwargs["reshard_after_forward"] = False  # Optimization from TorchTitan
            fully_shard([model.norm, model.output], **norm_output_kwargs)
    
    # Finally, shard the whole model
    fully_shard(model, **fsdp_kwargs)

def apply_tp(model, tp_mesh: DeviceMesh, tp_degree: int = 1, async_tp: bool = False):
    if local_rank == 0:
        print("Applying Tensor Parallel parallelization plan...")
    # Parallelize top-level modules
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # Parallelize each transformer block
    for layer_id, transformer_block in model.layers.named_children():
        layer_tp_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    
    if async_tp:
        if tp_degree <= 1:
            raise ValueError("tp_degree must be > 1 when async-tp is enabled")
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

def _apply_ac_to_transformer_block(
    module: nn.Module, mode: str, *, base_fqn: Optional[str] = None
):
    valid_ac_modes = ("full")
    if mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {mode}. Valid modes: {valid_ac_modes}"
        )

    if mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

def apply_activation_checkpointing(model, mode="full"):
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block, mode, base_fqn=f"layers.{layer_id}"
        )
        model.layers.register_module(layer_id, transformer_block)

def compile_loss(loss_function, backend="inductor"):
    return torch.compile(loss_function, backend=backend)


def get_train_context(enable_compiled_autograd: bool = False):
    """
    Returns a factory that produces a context manager for training steps, optionally
    nesting an external context (e.g., CP/TP).
      train_context = get_train_context(...)
      with train_context(optional_ctx):
          ...

    enable_compiled_autograd is accepted for API parity; it's unused here but can be
    hooked up later when integrating compiled autograd.
    """

    def train_context(optional_ctx=None):
        @contextmanager
        def _ctx():
            with ExitStack() as stack:
                # Enable compiled autograd if requested
                if enable_compiled_autograd:
                    stack.enter_context(
                        torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                    )
                if optional_ctx is not None:
                    stack.enter_context(optional_ctx)
                yield
        return _ctx()

    return train_context


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list,
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set,
    cp_rotate_method: str,
):
    """
    Thin wrapper over torch.distributed.tensor.experimental.context_parallel
    mirroring TorchTitan's utility. It also sets the rotate method to either
    'allgather' or 'alltoall'.
    """
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        raise RuntimeError(
            f"Your PyTorch version ({torch.__version__}) does not expose experimental Context Parallel API"
        )

    if cp_rotate_method not in ("allgather", "alltoall"):
        raise ValueError("cp_rotate_method must be 'allgather' or 'alltoall'")

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )

def evaluation(
    model,
    eval_dataloader,
    wandb,
    local_rank,
    loss_fn,
    device,
    train_context,
):
    if local_rank == 0:
        print("RUNNING EVAL")

    model.eval()
    losses = torch.tensor(0.0, device=device)
    for step, batch in enumerate(eval_dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
        position_ids = batch["position_ids"].to(device) if "position_ids" in batch else None
        with torch.no_grad():
            # Optional CP/TP context wrapper for future integration
            with train_context(None):
                logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)

        loss = loss_fn(logits, labels)
        losses += loss.float()

    losses = losses / (step + 1)
    # Under eval(), reduce across world (no TP/CP context here)
    val_loss = get_all_reduce_mean(losses.clone()).item()

    if local_rank == 0:
        print(f"Validation Loss {val_loss:.6f}")

    if local_rank == 0:
        wandb.log(
            {
                "val_loss": val_loss,
            }
        )

    return val_loss


def get_dataloader(
    max_length,
    dataset,
    dp_degree,
    dp_rank,
    shuffle,
    seed,
    collator,
    batch_size,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_degree,
        rank=dp_rank,
        shuffle=shuffle,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
    )

    return sampler, loader


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def should_run_eval(total_steps, times_to_run, current_step):
    return current_step % (total_steps // times_to_run) == 0


def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler, step_size):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
        },
    )

    current_loss = f"{loss_tensor:.6f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")
    pbar.update(step_size)


def get_all_reduce_mean(tensor, group=None):
    """
    Mean-reduce across the provided group; if None, reduce across the world.
    Use DP group for FSDP+TP and FSDP+CP. TP produces replicated scalars and
    CP loss is computed per-shard; averaging across DP is sufficient for stable
    metrics without over-reduction.
    """
    if group is not None:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
        denom = torch.distributed.get_world_size(group=group)
    else:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        denom = torch.distributed.get_world_size()
    tensor = tensor / denom
    return tensor

def dist_reduce_sum(tensor, mesh: DeviceMesh) -> float:
    """
    Distributed sum reduction
    Handles DTensor inputs properly and returns a float.
    """
    from torch.distributed._tensor import DTensor
    import torch.distributed._functional_collectives as funcol
    
    # Handle DTensor inputs
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    
    assert tensor.numel() == 1, f"Expected scalar tensor, got tensor with {tensor.numel()} elements"
    result = funcol.all_reduce(tensor, reduceOp="sum", group=mesh.get_group())
    return result.item()

def dist_reduce_mean(tensor, mesh: DeviceMesh) -> float:
    """
    Distributed mean reduction
    Handles DTensor inputs properly and returns a float.
    """
    from torch.distributed._tensor import DTensor
    import torch.distributed._functional_collectives as funcol
    
    # Handle DTensor inputs
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    
    assert tensor.numel() == 1, f"Expected scalar tensor, got tensor with {tensor.numel()} elements"
    result = funcol.all_reduce(tensor, reduceOp="avg", group=mesh.get_group())
    return result.item()

def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)

def get_decay_steps(num_training_steps, decay_ratio=0.25):
    return math.ceil(num_training_steps * decay_ratio)

def clip_model_gradients(model, max_grad_norm):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()

def clip_grad_norm_sharded(model, max_grad_norm: float, dp_mesh: DeviceMesh | None = None, tp_mesh: DeviceMesh | None = None) -> float:
    """
    Compute and apply global grad norm clipping when params/grads are sharded by FSDP2 (DP)
    and/or TP. Avoids stacking mixed-mesh DTensors by operating on local shards and
    all-reducing squared norms across DP and TP dimensions.

    Returns the global grad norm (float).
    """
    device = None
    local_sq = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        # If DTensor, operate on local shard for norm calculation
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if device is None:
            device = g_local.device
        # Accumulate squared L2 norms in fp32 for stability
        local_sq += g_local.detach().float().norm(2).pow(2)

    # Reduce across DP then TP to cover all unique shards
    if dp_mesh is not None:
        dp_group = dp_mesh.get_group()
        torch.distributed.all_reduce(local_sq, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    if tp_mesh is not None:
        tp_group = tp_mesh.get_group()
        torch.distributed.all_reduce(local_sq, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    total_norm = local_sq.sqrt().item()
    if max_grad_norm is None or max_grad_norm <= 0:
        return total_norm

    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        # Scale grads in-place consistently across shards
        for p in model.parameters():
            if p.grad is None:
                continue
            p.grad.mul_(clip_coef)

    return total_norm

@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm

def get_scheduler(local_rank, scheduler_type, optimizer, max_steps):
    warmup_steps = get_warmup_steps(max_steps)
    decay_steps = get_decay_steps(max_steps)

    if local_rank == 0 and scheduler_type != "warmup_stable_decay":
        print(f"[WARMUP STEPS]: {warmup_steps}")
        print(f"[MAX STEPS]: {max_steps}")
        print(f"[SCHEDULER]: {scheduler_type}")

    if local_rank == 0 and scheduler_type == "warmup_stable_decay":
        print(f"[WARMUP STEPS]: {warmup_steps}")
        print(f"[DECAY STEPS]: {decay_steps}")
        print(f"[STABLE STEPS]: {max_steps - decay_steps - warmup_steps}")
        print(f"[MAX STEPS]: {max_steps}")
        print(f"[SCHEDULER]: {scheduler_type}")

    if scheduler_type != "warmup_stable_decay":
        return transformers.get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        return transformers.get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_decay_steps=decay_steps,
            num_training_steps=max_steps,
            warmup_type="linear",
            decay_type="cosine"
        )


def save_model(local_rank, model, tokenizer, outpath, current_epoch, current_step, hf_config, model_args, *, use_async_dcp: bool = False, storage_writer: Optional[StorageWriter] = None, checkpoint_state: Optional[dict] = None):
    """
    Save checkpoints efficiently with DCP during training, and export HF once at the end.

    - For per-epoch saves: use DCP distributed save (no all-gather, non-blocking for rank0).
    - For the final save: aggregate full state on CPU and export HF as before.
    """
    # Final export path uses HF format; intermediate saves use DCP
    is_final_export = isinstance(current_step, str) and current_step == "final"

    if not is_final_export:
        # DCP distributed save (FSDP2-friendly). Avoid aggregating full state.
        # Build checkpoint directory and id
        ckpt_dir = f"{outpath}/epoch_{current_epoch}/step_{current_step}"
        os.makedirs(ckpt_dir, exist_ok=True)

        # Each rank provides its shard-aware state dict
        state = get_model_state_dict(model)
        # Exclude non-persistent buffers if present
        state.pop("freqs_cis", None)

        if use_async_dcp:
            # Ensure we do not queue multiple concurrent saves
            if checkpoint_state is not None and checkpoint_state.get("future") is not None:
                checkpoint_state["future"].result()
                checkpoint_state["future"] = None

            future = dcp.async_save(
                state,
                checkpoint_id=ckpt_dir,
                storage_writer=storage_writer,
            )
            if checkpoint_state is not None:
                checkpoint_state["future"] = future

            # Save tokenizer once (small and fast); no barrier for async path
            if local_rank == 0:
                tokenizer.save_pretrained(ckpt_dir)
            return
        else:
            # Save with DCP; all ranks participate
            dcp.save(state, checkpoint_id=ckpt_dir)

            # Save tokenizer once
            if local_rank == 0:
                tokenizer.save_pretrained(ckpt_dir)

            dist.barrier()
            return

    # Final HF export (training is done, aggregation cost is acceptable)
    # Wait for any in-flight async save before final export
    if use_async_dcp and checkpoint_state is not None and checkpoint_state.get("future") is not None:
        checkpoint_state["future"].result()
        checkpoint_state["future"] = None

    model.freqs_cis = model._precompute_freqs_cis()
    cpu_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    if local_rank == 0:
        print(f"SAVING FINAL HF MODEL")
        base = LlamaForCausalLM(hf_config)
        hf_state_dict = to_hf(cpu_state_dict, model_args)
        base.load_state_dict(hf_state_dict, strict=True)
        final_dir = f"{outpath}/epoch_{current_epoch}/step_{current_step}"
        os.makedirs(final_dir, exist_ok=True)
        base.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-dcp", dest="use_async_dcp", action="store_true", help="Enable async DCP checkpointing")
    parser.add_argument("--async-dcp-pinned", dest="use_pinned_writer", action="store_true", help="Use pinned-memory writer with async DCP")
    parser.add_argument("--tp-degree", type=int, default=1, help="Tensor parallel degree (1 disables TP)")
    parser.add_argument("--async-tp", action="store_true", help="Enable Async TP")
    parser.add_argument("--cp-degree", type=int, default=1, help="Context parallel degree (1 disables CP)")
    parser.add_argument(
        "--cp-rotate",
        type=str,
        default="allgather",
        choices=["allgather", "alltoall"],
        help="Context parallel rotate method: allgather or alltoall",
    )
    args, _ = parser.parse_known_args()

    # Checkpointing options (also configurable via CLI flags)
    use_async_dcp = bool(args.use_async_dcp)
    use_pinned_writer = bool(args.use_pinned_writer)
    async_tp = bool(args.async_tp)
    if use_pinned_writer and not use_async_dcp:
        # Pinned writer requires async mode; promote to async
        use_async_dcp = True
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    # Enable CPU backend if async DCP is requested (required by async_save)
    _backend = "cpu:gloo,cuda:nccl" if use_async_dcp else "nccl"
    dist.init_process_group(_backend, rank=local_rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    # Build device mesh for DP x TP x (optional) CP
    tp_degree = int(args.tp_degree)
    if tp_degree < 1:
        raise ValueError("--tp-degree must be >= 1")
    
    cp_degree = int(args.cp_degree)
    if cp_degree < 1:
        raise ValueError("--cp-degree must be >= 1")
    
    if world_size % (tp_degree * cp_degree) != 0:
        raise ValueError(f"WORLD_SIZE {world_size} must be divisible by tp-degree*cp-degree {tp_degree*cp_degree}")

    dp_degree = world_size // (tp_degree * cp_degree)
    
    # Validate dimensions
    assert dp_degree >= 1, f"dp_degree must be >= 1, got {dp_degree}"
    assert tp_degree >= 1, f"tp_degree must be >= 1, got {tp_degree}"
    assert cp_degree >= 1, f"cp_degree must be >= 1, got {cp_degree}"
    assert dp_degree * tp_degree * cp_degree == world_size, (
        f"Invalid parallel dims: dp({dp_degree}) * tp({tp_degree}) * cp({cp_degree}) != WORLD_SIZE({world_size})"
    )

    # Build structured mesh
    def build_device_mesh():
        dims = []
        names = []
        
        # Add dimensions that are > 1
        for d, name in zip([dp_degree, cp_degree, tp_degree], ["dp", "cp", "tp"]):
            if d > 1:
                dims.append(d)
                names.append(name)
        
        if local_rank == 0:
            print(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        
        mesh = init_device_mesh("cuda", dims, mesh_dim_names=names)
        
        # Create submeshes using flattening
        dp_mesh_dim_names = []
        dp_cp_mesh_dim_names = []
        
        # Add dp dimension if enabled
        if dp_degree > 1:
            dp_mesh_dim_names.append("dp")
            dp_cp_mesh_dim_names.append("dp")
        
        # Add cp dimension if enabled  
        if cp_degree > 1:
            dp_cp_mesh_dim_names.append("cp")
        
        # Create flattened submeshes
        if dp_mesh_dim_names:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp_flat")
        if dp_cp_mesh_dim_names:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        
        return mesh

    world_mesh = build_device_mesh()
    
    # Extract submeshes
    tp_enabled = tp_degree > 1
    cp_enabled = cp_degree > 1
    
    # Get submeshes using flattened names when possible
    if dp_degree > 1:
        dp_mesh = world_mesh["dp_flat"]
    else:
        dp_mesh = None
        
    if cp_enabled and dp_degree > 1:
        dp_cp_mesh = world_mesh["dp_cp"]
    elif cp_enabled:
        dp_cp_mesh = world_mesh["cp"]
    elif dp_degree > 1:
        dp_cp_mesh = world_mesh["dp_flat"] 
    else:
        dp_cp_mesh = None
        
    tp_mesh = world_mesh["tp"] if tp_enabled else None
    cp_mesh = world_mesh["cp"] if cp_enabled else None

    # Get dp rank for data loading  
    dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else 0
    
    # Calculate enabled flags
    dp_enabled = dp_degree > 1
    dp_cp_enabled = dp_enabled or cp_enabled
    
    cp_rotate_method = str(args.cp_rotate)

    if tp_enabled and local_rank == 0:
        print(f"TP DEGREE: {tp_degree}")
    if cp_enabled and local_rank == 0:
        print(f"CP ROTATE METHOD: {cp_rotate_method}")

    # Training context factory
    # Compiled autograd can hold references that conflict with FSDP2 param freeing.
    # Disable to avoid post-backward reshard/free errors under FSDP2+TP.
    if tp_enabled or cp_enabled:
        # Avoid compiled autograd with FSDP2 + TP/CP to prevent param freeing conflicts
        train_context = get_train_context(enable_compiled_autograd=False)
    else:
        train_context = get_train_context(enable_compiled_autograd=False)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    scheduler_type = "cosine"
    # scheduler_type = "warmup_stable_decay"
    seed = 877645  # set your seed
    transformers.set_seed(seed)

    date_of_run = current_timestamp_ist()
    notes = "llama32_3b_flash_attn_fsdp2_cp_torch_compile_dcp_deepseek_r1_sft"
    run_id = "exp_" + date_of_run + "_" + notes
    output_dir = f"/mnt/ssd2/shreyansh/models/llama32/{run_id}"
    max_length = 16384 # 12288  # adjust as needed
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True  # multipack sampler already does random sampling
    train_batch_size = 1 # adjust as needed
    validation_batch_size = 1  # adjust as needed
    epochs = int(args.epochs) if getattr(args, "epochs", None) else 5
    gradient_accumulation_steps = 4
    lr = 7e-05 # 5e-06  # adjust as needed
    weight_decay = 0.01  # adjust as needed
    gradient_clipping = 1.0  # adjust as needed
    train_on_inputs = False  # whether to train on instruction tokens
    packing = None # None, "ffd"
    compile = True
    use_flash_attn_api = False  # whether to use Flash Attention API module
    use_flash_attn_sdpa = True  # whether to use Flash Attention backend from SDPA

    if local_rank == 0:
        print(f"OUTPUT DIR: {output_dir}")
        print(f"USING FLASH ATTENTION (from API): {use_flash_attn_api}")
        print(f"USING FLASH ATTENTION (from SDPA): {use_flash_attn_sdpa}")
        os.makedirs(output_dir, exist_ok=True)

    if cp_enabled:
        assert (
            max_length % (2 * cp_degree) == 0
        ), f"max_length {max_length} must be divisible by 2*cp_degree={2*cp_degree} when CP is enabled"

    model, tokenizer, hf_config, model_args = setup_model(model_name, max_length, use_flash_attn_api, use_flash_attn_sdpa, tp_enabled)
    num_params = sum([p.numel() for p in model.parameters()])

    # Apply Tensor Parallel plan before compilation and FSDP wrapping
    if tp_enabled:
        apply_tp(model, tp_mesh=tp_mesh, tp_degree=tp_degree, async_tp=async_tp)

    if compile:
        torch._dynamo.config.capture_scalar_outputs = True
        backend = "inductor"
        compile_model(model, backend=backend, fullgraph=False)
    
    # Use the DP×CP flattened mesh for FSDP (TorchTitan-style dp_shard_cp),
    # so parameters are sharded across both dp_shard and cp dimensions.
    apply_fsdp(model, dp_mesh=dp_cp_mesh, tp_enabled=tp_enabled, cp_enabled=cp_enabled)

    # if tp_enabled:
    #     # Move freqs_cis to device and distribute after FSDP
    #     with torch.no_grad():
    #         # Ensure freqs_cis is properly distributed as replicated tensor
    #         if not isinstance(model.freqs_cis, DTensor):
    #             model.freqs_cis = distribute_tensor(model.freqs_cis.to(device), tp_mesh, [Replicate()])

    assert isinstance(model, FSDPModule)

    optimizer = get_optimizer(model, lr, weight_decay)

    # Prepare async DCP writer and state if requested
    checkpoint_state = {"future": None}
    storage_writer = None
    if use_async_dcp:
        storage_writer = StorageWriter(
            cache_staged_state_dict=use_pinned_writer,
            path=output_dir,
        )
    
    # train_ds = ["../data/sft/train/moonshot_kimi_k2_data_train_v2_sft_train_llama.jsonl"]
    # val_ds = ["../data/sft/train/moonshot_kimi_k2_data_val_v2_sft_val_llama.jsonl"]
    train_ds = ["../data/sft/train/deepseek_r1_data_train_sft_train_llama.jsonl"]
    val_ds = ["../data/sft/train/deepseek_r1_data_val_sft_val_llama.jsonl"]

    train_dataset = SupervisedDataset(train_on_inputs, tokenizer, train_ds, packing=packing)
    val_dataset = SupervisedDataset(train_on_inputs, tokenizer, val_ds, packing=packing)
    if packing == "ffd":
        assert use_flash_attn_api is True
        collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=True,
            padding_free=True,
            return_position_ids=True,
            pad_to_multiple_of=None,
        )
    else:
        collator = DataCollatorForSupervisedDataset(tokenizer, pad_to_multiple_of=8)

    train_sampler, train_loader = get_dataloader(
        max_length,
        train_dataset,
        dp_degree,
        dp_rank,
        shuffle,
        seed,
        collator,
        train_batch_size,
    )
    val_sampler, val_loader = get_dataloader(
        max_length,
        val_dataset,
        dp_degree,
        dp_rank,
        shuffle,
        seed,
        collator,
        validation_batch_size,
    )

    total_steps_per_epoch = len(train_loader)

    max_steps = total_steps_per_epoch * epochs / gradient_accumulation_steps
    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="combined_sft_llama32_3b_fsdp_v3_wordle",
            name=run_id,
            config={
                "model_name": model_name,
                "run_id": run_id,
                "date": date_of_run,
                "dataset_size": len(train_dataset),
                "dataset": ",".join(train_ds),
                "validation": ",".join(val_ds),
                "weight_decay": weight_decay,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "shuffle": shuffle,
                "seed": seed,
                "train_on_inputs": train_on_inputs,
                "epochs": epochs,
                "acc_steps": gradient_accumulation_steps,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
                "use_flash_attn_api": use_flash_attn_api,
                "use_flash_attn_sdpa": use_flash_attn_sdpa,
            },
        )

    if gradient_checkpointing and not (cp_enabled):
        apply_activation_checkpointing(model, "full")
        if local_rank == 0:
            print("Applying activation checkpointing")
    elif gradient_checkpointing and (cp_enabled) and local_rank == 0:
        print("Skipping activation checkpointing under TP/CP to avoid interactions with sharded params and checkpointed graphs")

    loss_fn = cross_entropy_loss
    loss_fn = compile_loss(loss_fn, backend="inductor")

    model.train()
    
    # Verify all parameters require gradients
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if local_rank == 0:
                print(f"WARNING: Parameter {name} does not require gradients")
    
    if local_rank == 0:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {param_count:,}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
    dist.barrier()
    train_iterator = iter(train_loader)
    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            total=total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )
        current_step = 0
        while True:
            training_loss_sum = torch.tensor(0.0, device=device)
            training_token_count = torch.tensor(0.0, device=device)
            actual_accumulation_steps = 0
            for acc_step in range(gradient_accumulation_steps):
                # Handle gradient sync properly for FSDP+TP
                model.require_backward_grad_sync = (acc_step == gradient_accumulation_steps - 1) or \
                                                        (current_step == total_steps_per_epoch - 1)
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                position_ids = batch["position_ids"].to(device) if "position_ids" in batch else None
                # Optional CP context
                optional_context_parallel_ctx = None
                if cp_enabled:
                    cp_buffers = [input_ids, labels, model.freqs_cis]
                    cp_seq_dims = [1, 1, 0]
                    # if attention_mask is not None:
                    #     cp_buffers.append(attention_mask)
                    #     cp_seq_dims.append(1)
                    # if position_ids is not None:
                    #     cp_buffers.append(position_ids)
                    #     cp_seq_dims.append(1)
                    optional_context_parallel_ctx = create_context_parallel_ctx(
                        cp_mesh=cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=cp_seq_dims,
                        cp_no_restore_buffers={input_ids, labels},
                        cp_rotate_method=cp_rotate_method,
                    )
                # forward under train_context (with optional CP)
                with train_context(optional_context_parallel_ctx):
                    logits = model(input_ids)
                    # Compute loss directly
                    loss = loss_fn(logits, labels)
                    # accumulate token-weighted loss for DP×CP logging
                    step_tokens = (labels[:, 1:] != -100).sum().to(device=device, dtype=torch.float32)
                    
                    # Handle case where CP shard has no valid tokens (all masked)
                    if step_tokens == 0:
                        # Create zero loss connected to model computation graph
                        # This ensures all CP ranks have consistent gradient flow
                        loss = (logits * 0.0).sum()  # Connected to model but evaluates to 0
                        contrib = torch.tensor(0.0, device=device, dtype=torch.float32)
                    else:
                        contrib = loss.detach() * step_tokens
                    
                    training_loss_sum += contrib
                    training_token_count += step_tokens
                    loss = loss / gradient_accumulation_steps

                    # backward inside the same context
                    loss.backward()
                
                # Clean up to reduce memory fragmentation and avoid storage issues
                del loss, logits
                
                # Force garbage collection for TP to avoid storage corruption
                if tp_enabled:
                    torch.cuda.empty_cache()
                    
                current_step += 1
                actual_accumulation_steps += 1

                if current_step == total_steps_per_epoch:
                    break

            # form DP×CP-reduced training metric (does not affect backward path)
            if dp_cp_enabled:
                training_loss_sum, training_token_count = dist_reduce_sum(
                    training_loss_sum, dp_cp_mesh
                ), dist_reduce_sum(
                    training_token_count, dp_cp_mesh
                )
                # if local_rank == 0:
                #     print(f"Training Loss Sum: {training_loss_sum}")
                #     print(f"Training Token Count: {training_token_count}")
                acc_loss_metric = (training_loss_sum / training_token_count)
            else:
                acc_loss_metric = (training_loss_sum / training_token_count).item()

            # clipping (global across dp_cp and tp meshes, like TorchTitan)
            if clip_gradients:
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=gradient_clipping, foreach=True)
                # grad_norm = clip_grad_norm_sharded(
                #     model,
                #     max_grad_norm=gradient_clipping,
                #     dp_mesh=dp_cp_mesh,
                #     tp_mesh=tp_mesh,
                # )
                # Check for NaN gradients
                if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                    if local_rank == 0:
                        print(f"WARNING: NaN/Inf gradient norm detected: {grad_norm}. Skipping optimizer step.")
                    # Skip optimizer step on NaN gradients
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            if local_rank == 0:
                log_stats(
                    pbar,
                    wandb,
                    round((current_step / total_steps_per_epoch), 2) + epoch,
                    acc_loss_metric,
                    grad_norm,
                    scheduler,
                    actual_accumulation_steps
                )

            if current_step == total_steps_per_epoch:
                # Run eval wrapped in CP if enabled (to maintain sequence sharding behavior)
                validation_loss_sum = torch.tensor(0.0, device=device)
                validation_token_count = torch.tensor(0.0, device=device)
                for step, batch in enumerate(val_loader):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                    position_ids = batch["position_ids"].to(device) if "position_ids" in batch else None
                    optional_context_parallel_ctx = None
                    if cp_enabled:
                        cp_buffers = [input_ids, labels, model.freqs_cis]
                        cp_seq_dims = [1, 1, 0]
                        # if attention_mask is not None:
                        #     cp_buffers.append(attention_mask)
                        #     cp_seq_dims.append(1)
                        # if position_ids is not None:
                        #     cp_buffers.append(position_ids)
                        #     cp_seq_dims.append(1)
                        optional_context_parallel_ctx = create_context_parallel_ctx(
                            cp_mesh=cp_mesh,
                            cp_buffers=cp_buffers,
                            cp_seq_dims=cp_seq_dims,
                            cp_no_restore_buffers={input_ids, labels},
                            cp_rotate_method=cp_rotate_method,
                        )
                    model.eval()
                    with torch.no_grad():
                        with train_context(optional_context_parallel_ctx):
                            logits = model(input_ids)
                    loss_val = loss_fn(logits, labels).float()
                    step_tokens = (labels[:, 1:] != -100).sum().to(device=device, dtype=torch.float32)
                    
                    # Handle empty CP shards in validation
                    if step_tokens == 0:
                        contrib = torch.tensor(0.0, device=device, dtype=torch.float32)
                    else:
                        contrib = loss_val.detach() * step_tokens
                    
                    validation_loss_sum += contrib
                    validation_token_count += step_tokens
                # Reduce metrics across DP×CP
                if dp_cp_enabled:
                    validation_loss_sum, validation_token_count = dist_reduce_sum(
                        validation_loss_sum, dp_cp_mesh
                    ), dist_reduce_sum(
                        validation_token_count, dp_cp_mesh
                    )
                    val_loss = (validation_loss_sum / validation_token_count)
                else:
                    val_loss = (validation_loss_sum / validation_token_count).item()
                if local_rank == 0:
                    print(f"Validation Loss {val_loss:.6f}")
                    wandb.log({"val_loss": val_loss})

                if not use_async_dcp:
                    dist.barrier()

                save_model(
                    local_rank,
                    model,
                    tokenizer,
                    output_dir,
                    current_epoch,
                    current_step,
                    hf_config,
                    model_args,
                    use_async_dcp=use_async_dcp,
                    storage_writer=storage_writer,
                    checkpoint_state=checkpoint_state,
                )

                if not use_async_dcp:
                    dist.barrier()

                model.train()
                break

    # save final model
    save_model(
        local_rank,
        model,
        tokenizer,
        output_dir,
        epochs,
        "final",
        hf_config,
        model_args,
        use_async_dcp=use_async_dcp,
        storage_writer=storage_writer,
        checkpoint_state=checkpoint_state,
    )

    # Cleanup distributed resources
    if local_rank == 0:
        print("Training completed. Cleaning up distributed resources...")
    
    dist.barrier()
    dist.destroy_process_group()
    
    if local_rank == 0:
        print("Process group destroyed successfully.")
