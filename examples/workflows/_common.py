"""Shared CLI and pipeline-loading helpers for the workflow scripts.

Every script in ``examples/workflows/`` mirrors one ComfyUI workflow JSON
from upstream ``ComfyUI-Trellis2/example_workflows/`` and exposes the
same parameters via command-line flags. This module centralises:

* default image lookup (uses ``reference/microsoft-trellis2/...`` if
  present, else requires ``--image``).
* the pipeline-loader call (lazy: only invoked by the runnable scripts).
* the ``NotImplementedStub`` helper used by scripts whose underlying
  feature (1024_cascade, 1536_cascade, MeshRefiner, MultiView, Projection,
  ReconViaGen, Voxel2Mesh+TexSlat, Hunyuan, Qwen) has not yet been
  ported to MLX.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from trellis2_mlx.pipeline import Trellis2ImageTo3DPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMAGE_DIR = REPO_ROOT / "reference/microsoft-trellis2/assets/example_image"


def pick_default_image() -> Path | None:
    """First WebP / PNG / JPG under the upstream ``example_image/`` dir, or None."""
    if not DEFAULT_IMAGE_DIR.is_dir():
        return None
    for ext in ("*.webp", "*.png", "*.jpg"):
        for p in sorted(DEFAULT_IMAGE_DIR.glob(ext)):
            return p
    return None


def add_common_args(p: argparse.ArgumentParser, default_output: str) -> None:
    """Attach the universal ``--image / --output / --seed`` flags."""
    p.add_argument("--image", type=Path, default=None, help="Input image path")
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / default_output,
        help=f"Output GLB path (default: {default_output})",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")


def resolve_image(image_arg: Path | None) -> tuple[Image.Image, Path]:
    """Load the input image, falling back to the upstream example dir."""
    image_path = image_arg or pick_default_image()
    if image_path is None:
        print(
            "error: no --image provided and no upstream example images found. "
            "pass --image <path>.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not image_path.exists():
        print(f"error: image not found at {image_path}", file=sys.stderr)
        raise SystemExit(2)
    img = Image.open(image_path)
    print(f"input: {image_path.name}  mode={img.mode}  size={img.size}")
    return img, image_path


def load_pipeline(seed: int, with_texture: bool) -> Trellis2ImageTo3DPipeline:
    """Construct the pipeline and report load time."""
    from trellis2_mlx.pipeline import Trellis2Config, Trellis2ImageTo3DPipeline

    print(f"loading pipeline (~20-40 s; texture={'on' if with_texture else 'off'})...")
    t0 = time.perf_counter()
    p = Trellis2ImageTo3DPipeline(Trellis2Config(seed=seed, with_texture=with_texture))
    print(f"  ready in {time.perf_counter() - t0:.1f} s")
    return p


def warn_mode_fallback(requested: str, supported: str = "512") -> None:
    """Loudly note that the upstream workflow uses a higher-res mode we don't have."""
    if requested == supported:
        return
    print()
    print(f"  ⚠  upstream workflow defaults to mode={requested!r} but trellis2-mlx")
    print(f"     currently only supports mode={supported!r}. Running at {supported}.")
    print("     Higher-res cascade modes are roadmapped — track progress at")
    print("     https://github.com/havokentity/trellis2-mlx")
    print()


NOT_IMPLEMENTED_HEADER = (
    "═══════════════════════════════════════════════════════════════════════\n"
    "  trellis2-mlx: workflow not yet implemented\n"
    "═══════════════════════════════════════════════════════════════════════"
)


def not_implemented_stub(
    parser: argparse.ArgumentParser,
    workflow_json: str,
    missing_features: list[str],
    upstream_nodes: list[str],
    notes: str = "",
) -> int:
    """Parse args so ``--help`` works, then print a structured 'missing feature' report."""
    parser.parse_args()
    print()
    print(NOT_IMPLEMENTED_HEADER)
    print(f"  workflow:  {workflow_json}")
    print("  status:    NOT YET PORTED to trellis2-mlx")
    print("═══════════════════════════════════════════════════════════════════════")
    print()
    print("This workflow requires features that have not been implemented yet:")
    for f in missing_features:
        print(f"  • {f}")
    print()
    print(f"Upstream ComfyUI nodes used: {', '.join(upstream_nodes)}")
    if notes:
        print()
        print(notes)
    print()
    print("What works today (runnable):")
    print("  • examples/workflows/simple.py")
    print("  • examples/workflows/low_poly.py")
    print("  • examples/workflows/only_mesh_simple.py")
    print("  • examples/workflows/only_mesh_advanced.py")
    print("  • examples/workflows/better_texture.py")
    print("  • examples/workflows/advanced.py")
    print("  • examples/workflows/using_qwen_rembg.py")
    print()
    print("Track progress: https://github.com/havokentity/trellis2-mlx")
    return 2


__all__ = [
    "DEFAULT_IMAGE_DIR",
    "REPO_ROOT",
    "add_common_args",
    "load_pipeline",
    "not_implemented_stub",
    "pick_default_image",
    "resolve_image",
    "warn_mode_fallback",
]
