"""Image → 3D mesh, end-to-end. The real deal.

Usage (from repo root, with the venv active):

    python examples/image_to_3d.py
    python examples/image_to_3d.py --image path/to/your.png
    python examples/image_to_3d.py --seed 42 --output ~/my_3d.glb

This runs the full TRELLIS.2 inference pipeline on Apple Silicon via MLX:

    PIL image
        ↓ (alpha-mask crop if RGBA, else used as-is)
    DINOv3-L feature encoder         ← Meta weights via transformers
        ↓ [B, 1029, 1024] tokens
    Stage 1 SS-DiT  (12 steps, CFG 7.5/0.7)
        ↓ [1, 8, 16, 16, 16] SS latent
    SS-VAE decoder + maxpool
        ↓ [L, 3] active voxel coords at 32³ latent grid
    Stage 2 SLAT shape DiT  (12 steps, CFG 7.5/0.5)
        ↓ [L, 32] denormalized shape latent
    SC-VAE shape decoder  (4 stages, 16× upsample)
        ↓ O-Voxel at 512³ output resolution
    Flexible Dual Grid mesh extraction
        ↓ vertices + triangles
    GLB export

The published checkpoints take ~10-15 GB of RAM after fp32 promotion; on
M4 Max 36 GB this fits. End-to-end wall-time on M4 Max with the current
unoptimized MLX path: roughly 1-3 minutes for the 512 pipeline. The
Metal flash-attention + sparse-conv kernels will tighten that
significantly once they land.

Caveats (today):

* Only the ``512`` pipeline type is wired up (32³ SLAT → 512³ output).
  1024 / 1024_cascade / 1536_cascade are next.
* No BiRefNet — for best results, give the script an RGBA image with a
  clean alpha. RGB images will still process, but the model was trained
  on background-removed inputs.
* No texture / material — geometry only. Material decoder lands once
  the shape-side path is fully validated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from trellis2_mlx.pipeline import Trellis2Config, Trellis2ImageTo3DPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "reference/microsoft-trellis2/assets/example_image"
DEFAULT_OUTPUT = REPO_ROOT / "image_to_3d_demo.glb"


def _pick_default_image() -> Path | None:
    """Pick the first available WebP from the upstream example_image/ dir."""
    if not DEFAULT_IMAGE.is_dir():
        return None
    for ext in ("*.webp", "*.png", "*.jpg"):
        for p in sorted(DEFAULT_IMAGE.glob(ext)):
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, default=None, help="Input image path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output GLB path")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    if args.image is None:
        args.image = _pick_default_image()
        if args.image is None:
            print(
                "error: no --image provided and no upstream example images found. "
                "pass --image <path>.",
                file=sys.stderr,
            )
            return 2
        print(f"using default image: {args.image}")

    if not args.image.exists():
        print(f"error: image not found at {args.image}", file=sys.stderr)
        return 2

    image = Image.open(args.image)
    print(f"input: {args.image.name}  mode={image.mode}  size={image.size}")

    print(
        "loading pipeline (this takes ~20-40 s — promoting bf16/fp16 → fp32 for ~5 GB of weights)..."
    )
    t0 = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline(Trellis2Config(seed=args.seed))
    print(f"  ready in {time.perf_counter() - t0:.1f} s")

    print("running pipeline (DINOv3 + SS-DiT 12 steps + SS-VAE + SLAT-DiT 12 steps + decode)...")
    t0 = time.perf_counter()
    result = pipeline.run(image)
    print(
        f"  done in {time.perf_counter() - t0:.1f} s  →  "
        f"{result.vertices.shape[0]} vertices, {result.faces.shape[0]} triangles  "
        f"@ {result.output_resolution}³  "
        f"(coarse_active={result.coarse_coords.shape[0]} → fine_active={result.active_coords.shape[0]})"
    )

    if result.faces.shape[0] == 0:
        print(
            "warning: zero triangles — generation collapsed. Try a different --seed "
            "or check that the input image has a clean alpha channel.",
            file=sys.stderr,
        )

    written = pipeline.export_glb(result, args.output)
    size_kb = written.stat().st_size / 1024
    print(f"wrote GLB: {written}  ({size_kb:.1f} KB)")
    print(
        "open in macOS Quick Look (spacebar in Finder), Blender, or "
        "https://gltf-viewer.donmccurdy.com/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
