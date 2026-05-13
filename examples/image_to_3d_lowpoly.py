"""Image → 3D low-poly mesh, game-ready GLB with target poly count.

Equivalent of the ComfyUI ``LowPoly.json`` workflow: runs the 512-mode
pipeline, then closes holes and decimates to a target face count using
``pymeshlab`` (quadric edge collapse — the same algorithm upstream uses
through their ``cumesh.simplify`` CUDA library, but CPU-side here).
Per-vertex colors from the material decoder are interpolated through
the decimation, so the low-poly output keeps the texture.

Usage (from repo root, with the venv active):

    python examples/image_to_3d_lowpoly.py
    python examples/image_to_3d_lowpoly.py --target-faces 5000
    python examples/image_to_3d_lowpoly.py --image my.png --target-faces 2000

The default 5000 faces matches the upstream LowPoly workflow's target.
Reasonable values:

  *   500 –  2000   tiny — silhouette only, fine for far-LOD billboards
  *  2000 –  8000   typical low-poly game-asset target
  *  8000 – 30000   mid-poly, holds more detail
  * 30000+          high-poly; just write the raw mesh instead

End-to-end wall-time on M4 Max for the default 5000-face run: roughly
the same as the textured pipeline (~70 s for generation) plus another
10–30 s for hole-fill + decimation, so call it ~90 s.

Caveats:

* Generation is still 512 mode. The high-detail 1024 / 1024_cascade /
  1536_cascade modes from the upstream workflows aren't wired up yet,
  so very fine geometric detail (e.g. inscriptions on a sword hilt)
  may be lost before simplification ever runs.
* No proper texture-image baking yet (would need UV unwrap + rasterize
  pass). For now the low-poly mesh carries the colors as per-vertex
  attributes — adequate at typical low-poly counts but obviously
  coarser than a 4K texture map at high poly counts.
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
DEFAULT_OUTPUT = REPO_ROOT / "image_to_3d_lowpoly.glb"


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
    parser.add_argument(
        "--target-faces",
        type=int,
        default=5000,
        help="Quadric decimation target face count (default 5000; LowPoly.json default).",
    )
    parser.add_argument(
        "--max-hole-size",
        type=int,
        default=30,
        help="Maximum hole perimeter (edges) to close during hole-fill (default 30).",
    )
    parser.add_argument(
        "--no-fill-holes",
        action="store_true",
        help="Skip the hole-fill step (default is to fill holes ≤ --max-hole-size).",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Skip the texture pipeline (geometry only — faster, smaller GLB).",
    )
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

    with_texture = not args.no_texture
    print(
        f"loading pipeline (~20-40 s; texture={'on' if with_texture else 'off'})..."
    )
    t0 = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline(
        Trellis2Config(seed=args.seed, with_texture=with_texture)
    )
    print(f"  ready in {time.perf_counter() - t0:.1f} s")

    print("running pipeline ...")
    t0 = time.perf_counter()
    result = pipeline.run(image)
    gen_seconds = time.perf_counter() - t0
    print(
        f"  generation: {gen_seconds:.1f} s  →  "
        f"{result.vertices.shape[0]:,} vertices, {result.faces.shape[0]:,} triangles  "
        f"@ {result.output_resolution}³"
    )

    if result.faces.shape[0] == 0:
        print(
            "warning: zero triangles — generation collapsed. Try a different --seed "
            "or check that the input image has a clean alpha channel.",
            file=sys.stderr,
        )

    print(
        f"exporting GLB (fill_holes={not args.no_fill_holes}, "
        f"target_faces={args.target_faces:,}) ..."
    )
    t0 = time.perf_counter()
    written = pipeline.export_glb(
        result,
        args.output,
        repair=True,
        fill_holes=not args.no_fill_holes,
        target_faces=args.target_faces,
        max_hole_size=args.max_hole_size,
        verbose=True,
    )
    print(f"  export+repair: {time.perf_counter() - t0:.1f} s")

    size_kb = written.stat().st_size / 1024
    print(f"wrote GLB: {written}  ({size_kb:.1f} KB)")
    print(
        "open in macOS Quick Look (spacebar in Finder), Blender, or "
        "https://gltf-viewer.donmccurdy.com/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
