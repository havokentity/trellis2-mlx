"""HF checkpoint → MLX state dict converter.

Implements ``PHASE0_SPEC.md §7``. Walks the safetensors files published at
``huggingface.co/microsoft/TRELLIS.2-4B``, maps each PyTorch parameter name to
its MLX counterpart, applies any required transposes (Linear weight layouts
match between MLX and PyTorch — pending §7.2 verification), and writes MLX
safetensors.

The source-of-truth name → shape mapping is captured in
``docs/weight-inventory.md`` during Phase 0; this module consumes it.
"""

from __future__ import annotations

from pathlib import Path


def convert_checkpoint(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Convert TRELLIS.2-4B HF safetensors at ``src_dir`` to MLX format at ``dst_dir``."""
    raise NotImplementedError("weight conversion lands in Phase 1 step 2 (post-inventory)")
