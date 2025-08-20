import random
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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
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

    return torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.size(-1)).float(), labels.reshape(-1)
    )

def load_model(model_path, model_args):
    with torch.device("meta"):
        model = Transformer(model_args)
    
    model = model.to_empty(device="cpu")
    state_dict = torch.load(f"{model_path}/consolidated.00.pth", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, assign=True)

    # Load freqs_cis separately
    with torch.no_grad():
        model.freqs_cis = model._precompute_freqs_cis()
    return model

def setup_model(model_name, max_length):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=hf_token,
    )
    config.use_cache = False

    model_name_original = "llama_3b_instruct"
    model_path = f"./{model_name_original}/original"
    model_config = f"{model_path}/params.json"
    with open(model_config, "r") as f:
        params = json.load(f)

    params['max_seq_len'] = 131072
    model_args = ModelArgs(**params)

    model = load_model(model_path, model_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        use_auth_token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer, config, model_args

def compile_model(model, backend="inductor", fullgraph=False):
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, backend=backend, fullgraph=fullgraph)
        model.layers.register_module(layer_id, transformer_block)

def apply_fsdp(model):
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
        "reshard_after_forward": True
    }

    for layer_id, transformer_block in model.layers.named_children():
        fully_shard(transformer_block, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

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

def evaluation(
    model,
    eval_dataloader,
    wandb,
    local_rank,
    loss_fn,
    device,
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
            logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)

        loss = loss_fn(logits, labels)
        losses += loss.float()

    losses = losses / (step + 1)
    val_loss = get_all_reduce_mean(losses.clone()).item()

    if local_rank == 0:
        print(f"Validation Loss {val_loss:.4f}")

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
    world_size,
    local_rank,
    shuffle,
    seed,
    collator,
    batch_size,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
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

    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")
    pbar.update(step_size)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)

def get_decay_steps(num_training_steps, decay_ratio=0.25):
    return math.ceil(num_training_steps * decay_ratio)

def clip_model_gradients(model, max_grad_norm):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()


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


def save_model(local_rank, model, tokenizer, outpath, current_epoch, current_step, hf_config, model_args):
    cpu_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
    if local_rank == 0:
        print(f"SAVING MODEL")
        base = LlamaForCausalLM(hf_config)
        hf_state_dict = to_hf(cpu_state_dict, model_args)
        base.load_state_dict(hf_state_dict, strict=True)
        outpath += f"/epoch_{current_epoch}/step_{current_step}"
        base.save_pretrained(outpath)
        tokenizer.save_pretrained(outpath)
    
    dist.barrier()  

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    scheduler_type = "cosine"
    # scheduler_type = "warmup_stable_decay"
    seed = 877645  # set your seed
    transformers.set_seed(seed)

    date_of_run = current_timestamp_ist()
    notes = "llama32_3b_fsdp_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft"
    run_id = "exp_" + date_of_run + "_" + notes
    output_dir = f"/mnt/ssd2/shreyansh/models/llama32/{run_id}"
    max_length = 12288  # adjust as needed
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True  # multipack sampler already does random sampling
    train_batch_size = 2 # adjust as needed
    validation_batch_size = 2  # adjust as needed
    epochs = 5  # adjust as needed
    gradient_accumulation_steps = 4
    acc_steps = 0  # TODO: not implemented here yet
    lr = 7e-05 # 5e-06  # adjust as needed
    weight_decay = 0.01  # adjust as needed
    gradient_clipping = 1.0  # adjust as needed
    train_on_inputs = False  # whether to train on instruction tokens
    packing = None # None, "ffd"
    compile = True

    if local_rank == 0:
        print(f"OUTPUT DIR: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    model, tokenizer, hf_config, model_args = setup_model(model_name, max_length)
    num_params = sum([p.numel() for p in model.parameters()])

    if compile:
        torch._dynamo.config.capture_scalar_outputs = True
        backend = "inductor"
        compile_model(model, backend=backend, fullgraph=False)
    
    apply_fsdp(model)

    assert isinstance(model, FSDPModule)

    optimizer = get_optimizer(model, lr, weight_decay)
    
    train_ds = ["../data/sft/train/moonshot_kimi_k2_data_train_v2_sft_train_llama.jsonl"]
    val_ds = ["../data/sft/train/moonshot_kimi_k2_data_val_v2_sft_val_llama.jsonl"]
    # train_ds = ["../data/sft/train/openai_gpt_oss-120b_data_sft_train.jsonl"]
    # val_ds = ["../data/sft/train/openai_gpt_oss-120b_data_sft_val.jsonl"]

    train_dataset = SupervisedDataset(train_on_inputs, tokenizer, train_ds, packing=packing)
    val_dataset = SupervisedDataset(train_on_inputs, tokenizer, val_ds, packing=packing)
    if packing == "ffd":
        assert model.config._attn_implementation == "flash_attention_2"
        collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=True,
            padding_free=True,
            return_position_ids=True,
            pad_to_multiple_of=None,
        )
    else:
        collator = DataCollatorForSupervisedDataset(tokenizer)

    train_sampler, train_loader = get_dataloader(
        max_length,
        train_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        train_batch_size,
    )
    val_sampler, val_loader = get_dataloader(
        max_length,
        val_dataset,
        world_size,
        local_rank,
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
            project="combined_sft_llama32_3b_fsdp_v2_wordle",
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
                "acc_steps": acc_steps,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
            },
        )

    if gradient_checkpointing:
        apply_activation_checkpointing(model, "full")

    loss_fn = cross_entropy_loss
    loss_fn = compile_loss(loss_fn, backend=backend)

    model.train()
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
            acc_loss = torch.tensor(0.0, device=device)
            actual_accumulation_steps = 0
            for acc_step in range(gradient_accumulation_steps):
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
                # forward
                logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
                loss = loss_fn(logits, labels) 
                acc_loss += loss.item()
                loss /= gradient_accumulation_steps

                # backward
                loss.backward()
                
                current_step += 1
                actual_accumulation_steps += 1

                if current_step == total_steps_per_epoch:
                    break

            acc_loss /= actual_accumulation_steps
            
            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # avg loss over all processes
            acc_loss = get_all_reduce_mean(acc_loss).item()

            if local_rank == 0:
                log_stats(
                    pbar,
                    wandb,
                    round((current_step / total_steps_per_epoch), 2) + epoch,
                    acc_loss,
                    grad_norm,
                    scheduler,
                    actual_accumulation_steps
                )

            if current_step == total_steps_per_epoch:
                validation_loss = evaluation(
                    model,
                    val_loader,
                    wandb,
                    local_rank,
                    loss_fn,
                    device,
                )

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
                )

                dist.barrier()

                model.train()
                break

    # save final model
    save_model(local_rank, model, tokenizer, output_dir, epochs, "final", hf_config, model_args)