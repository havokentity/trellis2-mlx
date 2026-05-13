"""Mirror of upstream ``Advanced.json``. Full advanced generator + UV rasterizer.

Upstream workflow:

  Trellis2MeshWithVoxelAdvancedGenerator  mode=1024_cascade
                                          full per-stage knobs
  Trellis2ReconstructMeshWithQuad         mode=1, res=1024
  Trellis2SimplifyMesh                    target_faces=500_000  Cumesh
  Trellis2FillHolesWithCuMesh             defaults
  Trellis2FillHolesWithMeshlib            defaults
  Trellis2UnWrapAndRasterizer             2048² texture atlas, OPAQUE alpha

trellis2-mlx implementation:

* Mode=1024_cascade NOW supported natively (since trellis2-mlx 0.2).
* No UV unwrap + rasterizer yet — texture is baked as per-vertex color.
* The double hole-fill chain (cumesh + meshlib) collapses to a single
  pymeshlab pass.
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
        prog="advanced", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser, "Advanced.glb")
    parser.add_argument("--mode", default="1024_cascade",
                        choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-cfg", type=float, default=6.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=6.5)
    parser.add_argument("--target-faces", type=int, default=500_000)
    parser.add_argument("--texture-resolution", type=int, default=2048,
                        help="UV atlas size when --uv-atlas is set.")
    parser.add_argument(
        "--uv-atlas", action="store_true",
        help="Bake per-vertex colors into a UV-unwrapped 2D texture atlas.",
    )
    args = parser.parse_args()

    pipeline_type = resolve_pipeline_type(args.mode)
    image, _ = resolve_image(args.image)
    pipeline = load_pipeline(args.seed, with_texture=True, pipeline_type=pipeline_type)

    print(f"running pipeline (mode={pipeline_type}) ...")
    t0 = time.perf_counter()
    result = pipeline.run(image)
    print(
        f"  generation: {time.perf_counter() - t0:.1f} s  →  "
        f"{result.vertices.shape[0]:,} verts, {result.faces.shape[0]:,} tris @ {result.output_resolution}³"
    )

    print(
        f"exporting GLB (fill_holes=True, target_faces={args.target_faces:,}, "
        f"uv_atlas={args.uv_atlas}{f' @ {args.texture_resolution}²' if args.uv_atlas else ''}) ..."
    )
    t0 = time.perf_counter()
    written = pipeline.export_glb(
        result, args.output,
        repair=True, fill_holes=True, target_faces=args.target_faces,
        uv_atlas=args.uv_atlas, texture_size=args.texture_resolution,
        verbose=True,
    )
    print(f"  export+repair: {time.perf_counter() - t0:.1f} s")
    print(f"wrote GLB: {written}  ({written.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
