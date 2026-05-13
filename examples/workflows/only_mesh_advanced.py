"""Mirror of upstream ``Only_Mesh_Advanced.json``. Geometry-only with advanced gen.

Upstream workflow:

  Trellis2MeshWithVoxelAdvancedGenerator  seed=*  mode=1024_cascade
                                          ss_steps=12  ss_cfg=6.5  ss_cfg_rescale=0.2
                                          slat_steps=12  slat_cfg=6.5  slat_cfg_rescale=0.2
                                          (no texture stage)
  Trellis2Remesh                          1024 res
  Trellis2SimplifyMesh                    target_faces=2_000_000
  Trellis2FillHolesWithMeshlib            defaults

trellis2-mlx implementation:

* Generator runs at mode=512 (1024_cascade not yet implemented). All
  the per-stage CFG/step knobs are honored where they map onto our
  sampler params; mode-1024+ knobs are warned-about but ignored.
* No texture pipeline (mesh only).
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
        prog="only_mesh_advanced",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, "OnlyMeshAdvanced.glb")
    parser.add_argument("--mode", default="1024_cascade",
                        choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-cfg", type=float, default=6.5)
    parser.add_argument("--ss-cfg-rescale", type=float, default=0.2)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=6.5)
    parser.add_argument("--slat-cfg-rescale", type=float, default=0.2)
    parser.add_argument(
        "--target-faces", type=int, default=2_000_000,
        help="Quadric decimation target (upstream default: 2_000_000)."
    )
    parser.add_argument(
        "--smooth-iterations", type=int, default=5,
        help="Taubin smoothing iterations after decimation (default 5; 0 to disable). "
             "Higher = smoother but more shape erosion.",
    )
    args = parser.parse_args()

    pipeline_type = resolve_pipeline_type(args.mode)
    print(
        f"  upstream-only sampler knobs (currently advisory): "
        f"ss=(steps={args.ss_steps}, cfg={args.ss_cfg}, rescale={args.ss_cfg_rescale}) "
        f"slat=(steps={args.slat_steps}, cfg={args.slat_cfg}, rescale={args.slat_cfg_rescale})"
    )

    image, _ = resolve_image(args.image)
    pipeline = load_pipeline(args.seed, with_texture=False, pipeline_type=pipeline_type)

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
        result, args.output,
        repair=True, fill_holes=True, target_faces=args.target_faces,
        smooth_iterations=args.smooth_iterations, verbose=True,
    )
    print(f"  export+repair: {time.perf_counter() - t0:.1f} s")
    print(f"wrote GLB: {written}  ({written.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
