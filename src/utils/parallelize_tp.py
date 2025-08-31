import torch
import torch.nn as nn

try:
    # PT2.3+ composable tensor parallel APIs
    from torch.distributed.tensor import Replicate, Shard, DTensor
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        PrepareModuleInput,
        parallelize_module,
    )
except Exception as e:
    raise ImportError(
        "Torch composable tensor parallel APIs are required. Please use PyTorch >= 2.3 with DTensor enabled."
    ) from e


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    *,
    loss_parallel: bool = False,
):
    """
    Apply tensor + sequence parallelism to a Llama-style Transformer model using
    PyTorch DTensor parallel APIs.

    Assumptions about module structure (matches this repo's `Transformer`):
      - model.tok_embeddings, model.norm, model.output exist
      - model.layers is an ordered mapping of transformer blocks
      - each block contains: attention_norm, attention.{wq,wk,wv,wo}, ffn_norm,
        and feed_forward.{w1,w2,w3}

    loss_parallel=False is recommended for simplicity because it keeps the final
    logits as local Tensors (not DTensors), which works seamlessly with the loss
    computation in the training script.
    """
    
    # Debug device mesh - simplified logging
    import os
    world_rank = int(os.environ.get("RANK", 0))
    tp_local_rank = tp_mesh.get_local_rank()
    
    if world_rank == 0:
        print(f"Applying TP with mesh: {tp_mesh.mesh.shape}, device_type: {tp_mesh.device_type}")
        print(f"Model device before TP: {next(model.parameters()).device}")

    # Root parallelization: shard embedding output over sequence dim, norm over sequence, and control output proj
    parallelize_module(
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
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Per-layer plan: SP on norms; attention/feed-forward linears are split row/col wise
    for layer_idx, (layer_name, transformer_block) in enumerate(model.layers.named_children()):
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
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
            parallelize_plan=layer_plan,
        )
    
    if world_rank == 0:
        print(f"TP applied successfully to {len(model.layers)} layers")


def assert_seq_length_divisible(seq_len: int, tp_degree: int, cp_degree: int) -> None:
    """
    Sequence Parallel (TP) and Context Parallel (CP) combine along sequence dim.
    To avoid uneven shards, enforce:
      - if cp_degree > 1: seq_len % (tp_degree * 2 * cp_degree) == 0
      - else:              seq_len % tp_degree == 0
    """
    if tp_degree <= 1:
        return
    if cp_degree > 1:
        divisor = tp_degree * 2 * cp_degree
    else:
        divisor = tp_degree
    if seq_len % divisor != 0:
        raise ValueError(
            f"Sequence length {seq_len} must be divisible by {divisor} (tp_degree={tp_degree}, cp_degree={cp_degree})."
        )


def unshard_tp_state_dict(model: nn.Module) -> dict:
    """Materialize a full, unsharded state_dict from a TP-parallelized model.

    This function walks the model's state_dict and, when encountering DTensors
    that are Sharded, gathers shards across the mesh dimension used by TP and
    returns a regular CPU state_dict with full tensors.
    """
    full_state = {}
    state = model.state_dict()
    for name, tensor in state.items():
        if isinstance(tensor, DTensor):
            # Move to CPU eagerly to reduce GPU pressure during gather
            dt = tensor
            placement = dt.placements
            # If tensor is replicated already, just materialize to local tensor
            if all(p.is_replicate() for p in placement):
                full_state[name] = dt.to_local().cpu()
                continue
            # Otherwise, gather across the sharded dimension(s)
            try:
                # This re-distributes to a fully replicated DTensor and then extracts local
                replicated = dt.redistribute(device_mesh=dt.device_mesh, placements=[Replicate() for _ in placement])
                full_state[name] = replicated.to_local().cpu()
            except Exception:
                # As a fallback, try local view
                full_state[name] = dt.to_local().cpu()
        else:
            full_state[name] = tensor.detach().cpu()
    return full_state


