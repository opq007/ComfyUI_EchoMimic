# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# This package is a *restored* dependency of the ComfyUI EchoMimic port.
# The upstream EchoMimicV3 (and Wan2.1) ship a `dist` package that wraps
# xfuser's sequence-parallelism (SP) helpers so the transformer can be sharded
# across multiple GPUs.  The ComfyUI port forgot to vendor this directory,
# which made `from .dist import ...` raise `ModuleNotFoundError` and broke the
# whole custom node at load time.
#
# It is intentionally import-safe everywhere:
#   * If `xfuser` is installed AND `torch.distributed` is actually initialized,
#     the real xfuser SP primitives are re-exported (multi-GPU inference).
#   * Otherwise it degrades to a single-GPU identity:  world size == 1,
#     rank == 0, and a group whose `all_gather` is the identity.  This is both
#     the ComfyUI default and safe on a machine with several GPUs but no
#     explicit SP/distributed init (each process just uses device 0).
#
# This keeps the module importable and the transformer numerically correct on
# every topology without forcing an xfuser dependency.

from __future__ import annotations

import functools
import os
import torch

__all__ = [
    "get_sequence_parallel_rank",
    "get_sequence_parallel_world_size",
    "get_sp_group",
    "xFuserLongContextAttention",
    "XFUSER_AVAILABLE",
]

try:  # pragma: no cover - exercised only on multi-GPU hosts with xfuser
    from xfuser.core.distributed import (  # type: ignore
        get_sp_group as _xf_get_sp_group,
        get_sequence_parallel_rank as _xf_get_rank,
        get_sequence_parallel_world_size as _xf_get_world,
    )
    from xfuser.core.long_ctx_attention import (  # type: ignore
        xFuserLongContextAttention,
    )

    XFUSER_AVAILABLE = True
except Exception:  # pragma: no cover - xfuser is optional in ComfyUI
    XFUSER_AVAILABLE = False
    _xf_get_sp_group = None
    _xf_get_rank = None
    _xf_get_world = None

    xFuserLongContextAttention = None  # imported but never invoked in this path


def _sp_active() -> bool:
    """True only when xfuser's SP state is usable in this process."""
    if not XFUSER_AVAILABLE:
        return False
    try:
        import torch.distributed as dist

        return dist.is_initialized() and dist.get_world_size() > 1
    except Exception:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# Sequence parallel group — identity operator for the single-GPU fallback.
# ---------------------------------------------------------------------------
class _IdentitySPGroup:
    """A stand-in for xfuser's SP group that does nothing useful.

    With world size == 1 the transformer never reaches the SP branch
    (`if self.sp_world_size > 1:` guards everywhere), so this object is only a
    safety net that provides the same interface (`all_gather`) without erroring.
    """

    def all_gather(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return x

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1


_IDENTITY_SP_GROUP = _IdentitySPGroup()


def _env_world_size() -> int:
    """Honor explicit multi-GPU config set via env by the port's launcher."""
    return max(int(os.environ.get("SEQUENCE_PARALLEL_WORLD_SIZE", "1") or "1"), 1)


@functools.wraps(_xf_get_world) if _xf_get_world else (lambda f: f)
def get_sequence_parallel_world_size() -> int:
    if _sp_active():
        return _xf_get_world()  # type: ignore[misc]
    return _env_world_size()


@functools.wraps(_xf_get_rank) if _xf_get_rank else (lambda f: f)
def get_sequence_parallel_rank() -> int:
    if _sp_active():
        return _xf_get_rank()  # type: ignore[misc]
    try:
        import torch.distributed as dist
    except Exception:  # pragma: no cover
        return 0
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_sp_group():
    if _sp_active():
        return _xf_get_sp_group()  # type: ignore[misc]
    return _IDENTITY_SP_GROUP