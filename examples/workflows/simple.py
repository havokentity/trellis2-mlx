"""Mirror of upstream ``Simple.json``.

Upstream workflow defaults (widgets_values from the JSON):

  Trellis2MeshWithVoxelGenerator   seed=12345  mode=1024_cascade  steps=12  cfg=12
                                    chunk=999999  ss_steps=1  slat_steps=32
                                    euler sampler
  Trellis2ReconstructMeshWithQuad  mode=1  res=1024  smooth=True  decimate=True
  Trellis2SimplifyMesh             target_faces=500_000  method=Cumesh
  Trellis2FillHolesWithMeshlib     defaults
  Trellis2UnWrapAndRasterizer      texture_size=2048  alpha=OPAQUE

trellis2-mlx implementation:

* Mode=1024_cascade supported natively. mode=1024 falls back to cascade.
* Texture is baked as per-vertex colors, not UV-unwrap + 2048² texture
  (UnWrap + Rasterizer is not yet implemented). Default poly count is
  500_000 to match upstream's simplify target.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import (
    add_common_args,
    load_pipeline,
    resolve_image,
    resolve_pipeline_type,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="simple", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser, "Simple.glb")
    parser.add_argument(
        "--mode",
        default="1024_cascade",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        help="Generator mode (upstream default: 1024_cascade). Cascade modes (1024_cascade, 1536_cascade) now supported.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=500_000,
        help="Quadric decimation target (upstream default: 500_000).",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Skip the texture pipeline (geometry only).",
    )
    args = parser.parse_args()

    pipeline_type = resolve_pipeline_type(args.mode)

    image, _ = resolve_image(args.image)
    pipeline = load_pipeline(args.seed, with_texture=not args.no_texture, pipeline_type=pipeline_type)

    print(f"running pipeline (mode={pipeline_type}) ...")
    t0 = time.perf_counter()
    result = pipeline.run(image)
    print(
        f"  generation: {time.perf_counter() - t0:.1f} s  →  "
        f"{result.vertices.shape[0]:,} verts, {result.faces.shape[0]:,} tris @ {result.output_resolution}³"
    )

    print(f"exporting GLB (fill_holes=True, target_faces={args.target_faces:,}) ...")
    t0 = time.perf_counter()
    written = pipeline.export_glb(
        result,
        args.output,
        repair=True,
        fill_holes=True,
        target_faces=args.target_faces,
        verbose=True,
    )
    print(f"  export+repair: {time.perf_counter() - t0:.1f} s")
    print(f"wrote GLB: {written}  ({written.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
