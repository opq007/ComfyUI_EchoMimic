# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Restored xfuser sequence-parallel attention wrapper.
# `enable_multi_gpus_inference()` monkey-patches each `block.self_attn.forward`
# with `usp_attn_forward`.  When xfuser SP is inactive (the ComfyUI default, or
# any single-GPU process on a multi-GPU host) this function transparently
# delegates to the transformer's native attention implementation, so monkey
# patching is a safe no-op until multi-GPU SP is explicitly enabled.

from __future__ import annotations

import torch

from . import (
    get_sequence_parallel_world_size,
    xFuserLongContextAttention,
)


def _native_forward(self, x, seq_lens, grid_sizes, freqs, dtype):
    """Call the module's stock (non-SP) forward implementation.

    ``self`` is the patched ``WanSelfAttention`` instance.  The monkey-patch
    only replaces an *instance* attribute, so the original class method still
    lives on the class MRO — walk that to recover it and delegate.
    """
    for base in type(self).__mro__:
        fwd = base.__dict__.get("forward")
        if fwd is not None and fwd is not usp_attn_forward:
            return fwd(self, x, seq_len, grid_sizes, freqs, dtype)
    raise AttributeError(
        f"{type(self).__name__} has no native forward to delegate to")


def usp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    """Drop-in replacement for ``WanSelfAttention.forward``.

    Bound to the attention module, so ``self`` is the ``WanSelfAttention``
    instance and ``x`` is the token tensor ``[B, L, N, C]``.  When multi-GPU
    SP is active it performs xfuser long-context attention; otherwise it
    falls back to the module's stock attention so monkey-patching is a safe
    no-op and results stay correct on a single GPU (or a multi-GPU host
    running ComfyUI without explicit SP initialization).
    """
    if (get_sequence_parallel_world_size() > 1
            and xFuserLongContextAttention is not None):
        return _usp_attn_forward_xfuser(
            self, x, seq_lens, grid_sizes, freqs, dtype)
    return _native_forward(self, x, seq_lens, grid_sizes, freqs, dtype)


def _usp_attn_forward_xfuser(self, x, seq_lens, grid_sizes, freqs,
                             dtype=torch.bfloat16):
    """xfuser-backed SP attention, faithful to Wan2.1 upstream."""
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    def half(t):
        return t if t.dtype in (torch.float16, torch.bfloat16) else t.to(dtype)

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)
    q = _rope_apply(q, grid_sizes, freqs)
    k = _rope_apply(k, grid_sizes, freqs)

    attn = xFuserLongContextAttention()
    out = attn(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size,
    )
    return self.o(out.flatten(2))


def _rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        out.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(out).float()