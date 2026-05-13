"""Mirror of upstream ``LowPoly.json``. Game-ready low-poly textured mesh.

Upstream workflow:

  Trellis2MeshWithVoxelAdvancedGenerator  seed=123456  mode=512  steps=12  cfg=6.5
  Trellis2Remesh                          512 res
  Trellis2SimplifyMesh                    target_faces=5_000
  Trellis2FillHolesWithMeshlib            defaults
  Trellis2MeshTexturing                   steps=12  cfg=3  rescale=0.2
                                          mesh_res=1024  tex_res=1024  alpha=OPAQUE
  Trellis2SmoothNormals                   default

trellis2-mlx implementation:

* Mode=512 is upstream default — we match it natively.
* Texture is vertex-color via the material decoder; upstream's
  ``Trellis2MeshTexturing`` does UV-unwrap + 1024² texture atlas
  (not yet implemented here). At 5_000 faces the visual difference
  is small.
* Smooth normals: pymeshlab computes per-vertex normals on the final
  decimated mesh; we author them into the GLB.

Same as the older ``examples/image_to_3d_lowpoly.py`` — this lives
under ``workflows/`` so it appears alongside the rest of the workflow
mirrors.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import add_common_args, load_pipeline, resolve_image


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="low_poly", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser, "LowPoly.glb")
    parser.add_argument(
        "--target-faces", type=int, default=5_000,
        help="Quadric decimation target (upstream default: 5_000)."
    )
    parser.add_argument(
        "--max-hole-size", type=int, default=30,
        help="Maximum hole perimeter (edges) to close.",
    )
    parser.add_argument(
        "--no-fill-holes", action="store_true",
        help="Skip the hole-fill step."
    )
    parser.add_argument(
        "--no-texture", action="store_true",
        help="Skip the texture pipeline (geometry only).",
    )
    parser.add_argument(
        "--uv-atlas", action="store_true",
        help="Bake per-vertex colors into a UV-unwrapped 2D texture atlas "
             "(proper PBR material, much better for downstream engines).",
    )
    parser.add_argument(
        "--texture-size", type=int, default=1024,
        help="UV atlas size in pixels (default: 1024).",
    )
    parser.add_argument(
        "--smooth-iterations", type=int, default=5,
        help="Taubin smoothing iterations after decimation (default 5; 0 to disable). "
             "Higher = smoother but more shape erosion.",
    )
    args = parser.parse_args()

    image, _ = resolve_image(args.image)
    pipeline = load_pipeline(args.seed, with_texture=not args.no_texture)

    print("running pipeline (mode=512) ...")
    t0 = time.perf_counter()
    result = pipeline.run(image)
    print(
        f"  generation: {time.perf_counter() - t0:.1f} s  →  "
        f"{result.vertices.shape[0]:,} verts, {result.faces.shape[0]:,} tris @ {result.output_resolution}³"
    )

    print(
        f"exporting GLB (fill_holes={not args.no_fill_holes}, "
        f"target_faces={args.target_faces:,}, "
        f"uv_atlas={args.uv_atlas}{f' @ {args.texture_size}²' if args.uv_atlas else ''}) ..."
    )
    t0 = time.perf_counter()
    written = pipeline.export_glb(
        result, args.output,
        repair=True,
        fill_holes=not args.no_fill_holes,
        target_faces=args.target_faces,
        smooth_iterations=args.smooth_iterations,
        max_hole_size=args.max_hole_size,
        uv_atlas=args.uv_atlas,
        texture_size=args.texture_size,
        verbose=True,
    )
    print(f"  export+repair: {time.perf_counter() - t0:.1f} s")
    print(f"wrote GLB: {written}  ({written.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
