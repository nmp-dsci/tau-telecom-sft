"""Block-wise attention for the Mac (M2 only): the same result as SDPA in a tenth of the memory.

PyTorch's SDPA on Apple's GPU (MPS) builds the whole attention table: measured on the M1 Pro,
11 GB at 8K tokens and about 23 GB at 10K, which swapped a 32 GB Mac on v1's 10K-token rows.
This runs SDPA on 1,024 queries at a time against only the keys they may see, and recomputes
each block in the backward pass (activation checkpointing), so one block's table is alive at a
time. CUDA's SDPA already does this in its own kernels, so the GPU configs keep `sdpa`.

Registered with transformers as `chunked_sdpa`; a config selects it with
`model_init: {attn_implementation: chunked_sdpa}`. `tests/test_train.py` checks the outputs and
gradients against plain SDPA.
"""

from __future__ import annotations

from typing import Any

NAME = "chunked_sdpa"
BLOCK = 1024


def _repeat_kv(x: Any, n_rep: int) -> Any:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _block(q: Any, k: Any, v: Any, mask: Any, scale: float | None) -> Any:
    from torch.nn import functional

    return functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def chunked_sdpa(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """transformers' attention-function signature: q [B, H, T, D], k and v [B, Hkv, T, D];
    returns [B, T, H, D]. Causal; a 4-D mask, when given, is sliced per block."""
    import torch
    from torch.utils.checkpoint import checkpoint

    n_rep = query.shape[1] // key.shape[1]
    key, value = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    t = query.shape[2]
    out = []
    for s in range(0, t, BLOCK):
        e = min(s + BLOCK, t)
        if attention_mask is not None and attention_mask.dim() == 4:
            mask = attention_mask[:, :, s:e, :e]
        else:
            mask = torch.ones(e - s, e, dtype=torch.bool, device=query.device).tril(diagonal=s)
        q, k, v = query[:, :, s:e], key[:, :, :e], value[:, :, :e]
        if torch.is_grad_enabled():
            out.append(checkpoint(_block, q, k, v, mask, scaling, use_reentrant=False))
        else:
            out.append(_block(q, k, v, mask, scaling))
    return torch.cat(out, dim=2).transpose(1, 2).contiguous(), None


def register() -> None:
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register(NAME, chunked_sdpa)
    AttentionMaskInterface.register(NAME, sdpa_mask)
