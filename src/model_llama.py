# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from utils.attn_mask_utils import _prepare_4d_causal_attention_mask

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from utils.fa_utils import _flash_attention_forward
    HF_FA_AVAILABLE = True
except:
    HF_FA_AVAILABLE = False

print(f"FLASH_ATTN_AVAILABLE: {FLASH_ATTN_AVAILABLE}")
print(f"HF_FA_AVAILABLE: {HF_FA_AVAILABLE}")

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def prepare_flash_attn_inputs(attention_mask: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device = None):
    """
    Prepare inputs for flash attention from attention_mask (fallback method).
    
    Args:
        attention_mask: (batch_size, seq_len) tensor with 1s for valid tokens, 0s for padding
        batch_size: batch size
        seq_len: sequence length
        device: torch device to create tensors on (fallback to 'cuda' if None)
        
    Returns:
        cu_seqlens: cumulative sequence lengths
        max_seqlen: maximum sequence length in the batch
        total_tokens: total number of valid tokens
    """
    if attention_mask is None:
        # No padding, all sequences are full length
        if device is None:
            device = 'cuda'  # default fallback
        cu_seqlens = torch.arange(0, batch_size * seq_len + 1, seq_len, 
                                  dtype=torch.int32, device=device)
        max_seqlen = seq_len
        total_tokens = batch_size * seq_len
    else:
        # Calculate actual sequence lengths from attention mask
        seq_lens = attention_mask.sum(dim=1).to(dtype=torch.int32)  # (batch_size,) ensure int32
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=attention_mask.device),
                               seq_lens.cumsum(0).to(dtype=torch.int32)], dim=0)
        max_seqlen = seq_lens.max().item()
        total_tokens = seq_lens.sum().item()
    
    return cu_seqlens, max_seqlen, total_tokens


def detect_packed_sequences_from_position_ids(position_ids: torch.Tensor):
    """
    Detect packed sequences within each batch element using position_ids.
    
    Args:
        position_ids: (batch_size, seq_len) tensor with position indices
        
    Returns:
        valid_token_mask: (batch_size, seq_len) mask with True for valid tokens
        all_seq_lens: list of individual sequence lengths across all batch elements
    """
    batch_size, seq_len = position_ids.shape
    valid_token_mask = torch.zeros_like(position_ids, dtype=torch.bool)
    all_seq_lens = []
    
    for batch_idx in range(batch_size):
        pos_ids = position_ids[batch_idx]  # Shape: (seq_len,)
        
        # Find all sequences within this batch element
        current_seq_start = 0
        current_seq_lens = []
        
        for token_idx in range(seq_len):
            pos_val = pos_ids[token_idx].item()
            
            # Check if this is the start of a new sequence (position resets to 0)
            if token_idx > 0 and pos_val == 0 and pos_ids[token_idx-1].item() > 0:
                # End previous sequence
                seq_length = token_idx - current_seq_start
                if seq_length > 0:
                    current_seq_lens.append(seq_length)
                    valid_token_mask[batch_idx, current_seq_start:token_idx] = True
                current_seq_start = token_idx
            
            # Check for padding (non-incrementing values indicating end)
            elif token_idx > current_seq_start:
                expected_pos = token_idx - current_seq_start
                if pos_val != expected_pos:
                    # Check if this is consistent padding (same value repeated)
                    is_padding = False
                    if token_idx < seq_len - 1:
                        # Look ahead to see if values are repeating (padding pattern)
                        next_vals = pos_ids[token_idx:min(token_idx+3, seq_len)]
                        if len(next_vals) >= 2 and (next_vals == pos_val).all():
                            is_padding = True
                    
                    if is_padding:
                        # End current sequence at padding start
                        seq_length = token_idx - current_seq_start
                        if seq_length > 0:
                            current_seq_lens.append(seq_length)
                            valid_token_mask[batch_idx, current_seq_start:token_idx] = True
                        break
        
        # Handle the last sequence if it reaches the end
        if current_seq_start < seq_len:
            # Check if remaining tokens form a valid incrementing sequence
            remaining_length = 0
            for check_idx in range(current_seq_start, seq_len):
                expected = check_idx - current_seq_start
                if pos_ids[check_idx].item() == expected:
                    remaining_length += 1
                else:
                    break
            
            if remaining_length > 0:
                current_seq_lens.append(remaining_length)
                valid_token_mask[batch_idx, current_seq_start:current_seq_start + remaining_length] = True
        
        # Add this batch element's sequences to the global list
        all_seq_lens.extend(current_seq_lens)
    
    return valid_token_mask, all_seq_lens


def prepare_flash_attn_inputs_from_position_ids(position_ids: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device = None):
    """
    Prepare inputs for flash attention from position_ids, handling packed sequences.
    
    Supports packed sequences within batch elements:
    - position_ids = [0,1,2,3,0,1,2,0,1,2,3] represents 3 sequences of lengths [4,3,4]
    - Each individual sequence is processed separately with proper causal masking
    
    Args:
        position_ids: (batch_size, seq_len) tensor with position indices
        batch_size: batch size
        seq_len: sequence length
        device: torch device to create tensors on (fallback to 'cuda' if None)
        
    Returns:
        cu_seqlens: cumulative sequence lengths for ALL individual sequences
        max_seqlen: maximum individual sequence length
        total_tokens: total number of valid tokens across all sequences
        valid_token_mask: mask indicating which tokens are valid (not padding)
    """
    if position_ids is None:
        # No padding, all sequences are full length (no packing)
        if device is None:
            device = 'cuda'  # default fallback
        cu_seqlens = torch.arange(0, batch_size * seq_len + 1, seq_len, 
                                  dtype=torch.int32, device=device)
        max_seqlen = seq_len
        total_tokens = batch_size * seq_len
        valid_token_mask = None
    else:
        # Detect packed sequences within each batch element
        valid_token_mask, all_seq_lens = detect_packed_sequences_from_position_ids(position_ids)
        
        # Handle case where no sequences were detected (fallback)
        if not all_seq_lens:
            # Treat each batch element as single full sequence
            all_seq_lens = [seq_len] * batch_size
            valid_token_mask.fill_(True)
        
        # Create cumulative sequence lengths for ALL individual sequences
        seq_lens_tensor = torch.tensor(all_seq_lens, dtype=torch.int32, device=position_ids.device)
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=position_ids.device),
                               seq_lens_tensor.cumsum(0).to(dtype=torch.int32)], dim=0)
        max_seqlen = seq_lens_tensor.max().item() if len(all_seq_lens) > 0 else seq_len
        total_tokens = seq_lens_tensor.sum().item() if len(all_seq_lens) > 0 else batch_size * seq_len
    
    return cu_seqlens, max_seqlen, total_tokens, valid_token_mask


def pack_for_flash_attn_with_position_ids(tensor: torch.Tensor, valid_token_mask: torch.Tensor | None):
    """
    Pack tensor by removing padding tokens for flash attention using position-based mask.
    
    Args:
        tensor: (batch_size, seq_len, ...) tensor to pack
        valid_token_mask: (batch_size, seq_len) mask with True for valid tokens
        
    Returns:
        packed tensor with shape (total_valid_tokens, ...)
    """
    if valid_token_mask is None:
        # No padding, just flatten first two dimensions
        return tensor.view(-1, *tensor.shape[2:])
    else:
        # Only keep valid tokens
        return tensor[valid_token_mask]


def debug_position_ids(position_ids: torch.Tensor, sample_idx: int = 0):
    """
    Debug utility to print position_ids pattern for analysis and packed sequences.
    
    Args:
        position_ids: (batch_size, seq_len) position tensor
        sample_idx: which sequence in batch to debug (default: 0)
    """
    if position_ids is None:
        print("position_ids is None")
        return
    
    pos_seq = position_ids[sample_idx].cpu().tolist()
    print(f"Position IDs for batch element {sample_idx}: {pos_seq}")
    
    # Detect packed sequences within this batch element
    sequences = []
    current_seq_start = 0
    
    for j in range(1, len(pos_seq)):
        # Reset to 0 indicates new sequence
        if pos_seq[j] == 0 and pos_seq[j-1] > 0:
            seq_len = j - current_seq_start
            sequences.append((current_seq_start, j-1, seq_len, pos_seq[current_seq_start:j]))
            current_seq_start = j
        # Non-incrementing indicates padding/end
        elif pos_seq[j] != pos_seq[j-1] + 1 and j > current_seq_start + 1:
            # Check if this is consistent padding
            if j < len(pos_seq) - 1 and pos_seq[j] == pos_seq[j+1]:
                seq_len = j - current_seq_start
                sequences.append((current_seq_start, j-1, seq_len, pos_seq[current_seq_start:j]))
                break
    
    # Handle the last sequence
    if current_seq_start < len(pos_seq):
        seq_len = len(pos_seq) - current_seq_start
        # Check if it's a valid sequence (incrementing)
        valid = True
        for k in range(current_seq_start, len(pos_seq)):
            expected = k - current_seq_start
            if pos_seq[k] != expected:
                seq_len = k - current_seq_start
                valid = False
                break
        if seq_len > 0:
            end_idx = current_seq_start + seq_len - 1
            sequences.append((current_seq_start, end_idx, seq_len, pos_seq[current_seq_start:current_seq_start + seq_len]))
    
    print(f"Found {len(sequences)} packed sequences:")
    for i, (start, end, length, seq) in enumerate(sequences):
        print(f"  Sequence {i+1}: positions {start}-{end}, length={length}, pos_ids={seq}")


def debug_flash_attn_inputs(position_ids: torch.Tensor, cu_seqlens: torch.Tensor, valid_token_mask: torch.Tensor):
    """
    Debug utility to visualize flash attention inputs for packed sequences.
    
    Args:
        position_ids: (batch_size, seq_len) position tensor
        cu_seqlens: cumulative sequence lengths tensor
        valid_token_mask: (batch_size, seq_len) boolean mask
    """
    print("Flash Attention Debug Info:")
    print(f"cu_seqlens: {cu_seqlens.cpu().tolist()}")
    print(f"Number of sequences: {len(cu_seqlens) - 1}")
    print(f"Total valid tokens: {valid_token_mask.sum().item()}")
    print(f"Max sequence length: {(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()}")
    
    # Show which tokens are valid
    batch_size, seq_len = position_ids.shape
    for batch_idx in range(min(batch_size, 2)):  # Show first 2 batch elements
        valid_positions = valid_token_mask[batch_idx].nonzero().flatten().cpu().tolist()
        pos_vals = position_ids[batch_idx][valid_token_mask[batch_idx]].cpu().tolist()
        print(f"Batch {batch_idx} valid tokens at positions {valid_positions} with pos_ids {pos_vals}")


def test_packed_sequence_detection():
    """
    Test function to validate packed sequence detection with various examples.
    Useful for debugging position_ids patterns.
    """
    print("Testing packed sequence detection...")
    
    # Test case 1: Simple packed sequences
    test_cases = [
        # Case 1: [0,1,2,3,0,1,2,0,1,2,3] - 3 sequences of lengths [4,3,4]
        [0,1,2,3,0,1,2,0,1,2,3],
        # Case 2: [0,1,2,0,1,2,3,4,5] - 2 sequences of lengths [3,6] 
        [0,1,2,0,1,2,3,4,5],
        # Case 3: [0,1,2,3,4,0,0,0,0] - 1 sequence of length 5, then padding
        [0,1,2,3,4,0,0,0,0],
        # Case 4: [0,1,2,3,4,5,6,7,8] - 1 full sequence of length 9
        [0,1,2,3,4,5,6,7,8],
    ]
    
    for i, pos_list in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {pos_list}")
        # Create dummy position_ids tensor
        position_ids = torch.tensor([pos_list], dtype=torch.long)  # batch_size=1
        debug_position_ids(position_ids, sample_idx=0)
        
        # Test the actual function
        cu_seqlens, max_seqlen, total_tokens, valid_token_mask = prepare_flash_attn_inputs_from_position_ids(
            position_ids, batch_size=1, seq_len=len(pos_list), device=position_ids.device
        )
        print(f"  Result: cu_seqlens={cu_seqlens.tolist()}, max_seqlen={max_seqlen}, total_tokens={total_tokens}")


# Uncomment the line below to run tests for packed sequence detection:
# test_packed_sequence_detection()

# Usage Examples:
# 
# For regular training (non-packed sequences):
# - position_ids = [[0,1,2,3,4], [0,1,2,0,0]] → 2 sequences: [5, 3] lengths
#
# For packed sequences within batch elements:
# - position_ids = [[0,1,2,3,0,1,2,0,1,2,3]] → 3 sequences: [4,3,4] lengths packed in 1 batch element
#
# Flash Attention will automatically detect and handle both cases correctly.


class FlashAttention(nn.Module):
    """
    Multi-head attention module using HuggingFace's optimized Flash Attention utilities.
    
    Key Features:
    - Uses HuggingFace's battle-tested _flash_attention_forward for maximum performance
    - Automatically handles packed sequences and padding via position_ids/attention_mask
    - Optimized tensor operations and memory management
    - Proper torch.compile compatibility
    - Fallback to custom implementation if HF utilities unavailable
    
    Performance Benefits:
    - Much faster than custom Flash Attention implementations
    - Optimized packing/unpacking with minimal overhead
    - Efficient cu_seqlens calculation and sequence detection
    - Proper handling of edge cases and different sequence configurations
    
    Packed Sequence Support:
    - position_ids = [0,1,2,3,0,1,2,0,1,2,3] → 3 sequences: [4,3,4] lengths
    - HuggingFace utilities automatically detect and handle packed sequences
    - Supports both attention_mask and position_ids based sequence detection
    
    Args:
        model_args (TransformerModelArgs): Model configuration arguments.
    
    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.  
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.
        scale (float): Scaling factor for attention.
    
    Requirements:
        - position_ids MUST be provided (cannot be None)
        - attention_mask is completely ignored
    """
    
    def __init__(self, model_args):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("Flash Attention is not available. Please install flash-attn.")
            
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        """
        Forward pass using HuggingFace's optimized Flash Attention utilities.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            attention_mask (torch.Tensor, optional): Attention mask for padding (HF FA can handle this).
            position_ids (torch.LongTensor, optional): Position indices for packed sequences.
            
        Returns:
            torch.Tensor: Output tensor after attention.
            
        Note:
            This implementation uses HuggingFace's _flash_attention_forward which:
            - Automatically handles packed sequences from position_ids
            - Efficiently manages packing/unpacking operations
            - Properly calculates cu_seqlens for Flash Attention
            - Is optimized and battle-tested across many models
            - Has proper torch.compile compatibility
        """
        # Check if HF FA utilities are available, fallback to custom implementation
        if not HF_FA_AVAILABLE:
            # Fallback to our previous implementation if HF FA utilities not available
            return self._fallback_forward(x, freqs_cis, attention_mask, position_ids)
            
        bs, seqlen, _ = x.shape
        
        # QKV projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Reshape for multi-head attention
        xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seqlen, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # Repeat k/v heads if n_kv_heads < n_heads (GQA)
        # keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        # values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        
        # Use HuggingFace's optimized Flash Attention forward
        # This handles all the complex logic including:
        # - Packed sequence detection from position_ids
        # - Efficient packing/unpacking with minimal overhead
        # - Proper cu_seqlens calculation for Flash Attention
        # - Optimal tensor operations and memory management
        # - torch.compile compatibility

        output = _flash_attention_forward(
            query_states=xq,
            key_states=xk, 
            value_states=xv,
            attention_mask=attention_mask,  # HF FA can handle both mask and position_ids
            query_length=seqlen,
            is_causal=True,
            dropout=0.0,
            position_ids=position_ids,
            softmax_scale=self.scale,
            target_dtype=x.dtype,
        )
        
        # Reshape for output projection: (bs, seqlen, n_heads, head_dim) -> (bs, seqlen, dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)
    
    @torch._dynamo.disable
    def _fallback_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        """
        Fallback implementation using our custom Flash Attention logic.
        Only used if HuggingFace FA utilities are not available.
        """
        bs, seqlen, _ = x.shape
        
        # QKV projections 
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Reshape for multi-head attention
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        
        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # Repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        
        # Flash Attention requires position_ids for sequence detection
        if position_ids is None:
            raise ValueError(
                "FlashAttention requires position_ids to be provided. "
                "Please ensure position_ids are passed to the model."
            )
        
        # Use our custom sequence detection and packing logic
        cu_seqlens, max_seqlen, total_tokens, valid_token_mask = prepare_flash_attn_inputs_from_position_ids(
            position_ids, bs, seqlen, x.device
        )
        
        # Pack tensors by removing padding tokens 
        q_packed = pack_for_flash_attn_with_position_ids(xq, valid_token_mask)
        k_packed = pack_for_flash_attn_with_position_ids(keys, valid_token_mask)
        v_packed = pack_for_flash_attn_with_position_ids(values, valid_token_mask)
        
        # Flash attention call
        output_packed = flash_attn_varlen_func(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
        )
        
        # Unpack output back to (batch_size, seq_len, n_heads, head_dim)
        if valid_token_mask is None:
            output = output_packed.view(bs, seqlen, self.n_heads, self.head_dim)
        else:
            output = torch.zeros(bs, seqlen, self.n_heads, self.head_dim, 
                               dtype=output_packed.dtype, device=output_packed.device)
            output[valid_token_mask] = output_packed
        
        # Reshape for output projection
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

class FlashAttentionSDPA(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.LongTensor): The position ids.
        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        # keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        if attention_mask is not None:
            # Attention mask was made 4D because the `attn_weights` above is 4D.
            # We probably can make this mask smarter if we want to pack sequences
            # together, instead of using padding. This optimization can be used in
            # inference. For training, if we want to pack sequences, data loader
            # will pass in a mask containing such info.
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,  # None, or user provided mask in 2D
                (bs, seqlen),
                x,
                0,  # past_key_values_length, 0 when training
            ).to(x.dtype)
            if attention_mask.size() != (bs, 1, seqlen, seqlen):
                raise ValueError(
                    f"Attention mask should be of size {(bs, 1, seqlen, seqlen)}, but is {attention_mask.size()}"
                )

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=xq.shape[-1] ** (-0.5),
                enable_gqa=True,
            )

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.LongTensor): The position ids.
        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        # keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        if attention_mask is not None:
            # Attention mask was made 4D because the `attn_weights` above is 4D.
            # We probably can make this mask smarter if we want to pack sequences
            # together, instead of using padding. This optimization can be used in
            # inference. For training, if we want to pack sequences, data loader
            # will pass in a mask containing such info.
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,  # None, or user provided mask in 2D
                (bs, seqlen),
                x,
                0,  # past_key_values_length, 0 when training
            )
            if attention_mask.size() != (bs, 1, seqlen, seqlen):
                raise ValueError(
                    f"Attention mask should be of size {(bs, 1, seqlen, seqlen)}, but is {attention_mask.size()}"
                )

        output = torch.nn.functional.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=xq.shape[-1] ** (-0.5),
            enable_gqa=True,
        )

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args, use_flash_attn_api: bool = False, use_flash_attn_sdpa: bool = False):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        
        # Choose attention implementation
        if use_flash_attn_api:
            self.attention = FlashAttention(model_args)
        elif use_flash_attn_sdpa:
            self.attention = FlashAttentionSDPA(model_args)
        else:
            self.attention = Attention(model_args)
            
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.LongTensor): The position ids.
        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis=freqs_cis, attention_mask=attention_mask, position_ids=position_ids)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (Linear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args, use_flash_attn_api: bool = False, use_flash_attn_sdpa: bool = False):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.use_flash_attn_api = use_flash_attn_api
        self.use_flash_attn_sdpa = use_flash_attn_sdpa

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args, use_flash_attn_api, use_flash_attn_sdpa)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            input_ids (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.LongTensor): The position ids.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(input_ids) if self.tok_embeddings else input_ids

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_mask, position_ids)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output