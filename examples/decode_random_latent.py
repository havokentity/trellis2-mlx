"""Run the SC-VAE shape decoder on a random latent and export a GLB.

Usage (from repo root, with the project venv active):

    python examples/decode_random_latent.py
    python examples/decode_random_latent.py --output ~/shape.glb --seed 42
    python examples/decode_random_latent.py --coarse-res 8 --n-voxels 32

Notes
-----
This is **not** an image-to-3D pipeline yet — it feeds **random noise** into
the SC-VAE shape decoder. The output is a real 3D mesh (you can open the
resulting ``.glb`` in any glTF viewer such as macOS Quick Look, Blender,
or https://gltf-viewer.donmccurdy.com/) but the geometry will look like
noise because there's no real latent driving it.

To get an actual generation you need the three DiT generators on top —
those land in Phase 1 steps 8–10 (see ``PHASE0_SPEC.md §9``).

Performance
-----------
First run reads ~948 MB of fp16 weights from disk; subsequent runs are
faster if the OS file cache is warm. The decoder itself runs in <1s for
the small active sets used here.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from trellis2_mlx.models.vae import ShapeDecoder
from trellis2_mlx.ovoxel.mesh_extract import extract_mesh
from trellis2_mlx.ovoxel.postprocess import export_glb
from trellis2_mlx.utils.weight_convert import shape_decoder_from_pt_state_dict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECODER = REPO_ROOT / "reference/weights/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
DEFAULT_OUTPUT = REPO_ROOT / "shape_decoder_demo.glb"


def _safetensors_load_all(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode())
        data_start = 8 + n
        header.pop("__metadata__", None)
        dtype_map = {"F16": np.float16, "F32": np.float32}
        out: dict[str, np.ndarray] = {}
        for k, info in header.items():
            dt = dtype_map.get(info["dtype"])
            if dt is None:
                raise ValueError(f"unsupported dtype {info['dtype']} for {k}")
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            buf = f.read(end - start)
            arr = np.frombuffer(buf, dtype=dt).reshape(info["shape"]).copy()
            out[k] = arr.astype(np.float32)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=DEFAULT_DECODER,
        help=f"Path to shape_dec_next_dc_f16c32_fp16.safetensors (default: {DEFAULT_DECODER})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output GLB path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--coarse-res",
        type=int,
        default=4,
        help="Coarse latent grid resolution. Output is 16× this (default: 4 → 64³).",
    )
    parser.add_argument(
        "--n-voxels",
        type=int,
        default=8,
        help="Number of active voxels in the random latent (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    args = parser.parse_args()

    if not args.decoder.exists():
        print(f"error: decoder weights not found at {args.decoder}", file=sys.stderr)
        print(
            'Download them first with: python -c "from huggingface_hub import snapshot_download;'
            " snapshot_download('microsoft/TRELLIS.2-4B', local_dir='reference/weights')\"",
            file=sys.stderr,
        )
        return 2

    print(f"loading decoder weights from {args.decoder} ...")
    t0 = time.perf_counter()
    state = _safetensors_load_all(args.decoder)
    decoder = ShapeDecoder()
    decoder.load_weights(shape_decoder_from_pt_state_dict(state))
    print(f"  {len(state)} parameter tensors loaded in {time.perf_counter() - t0:.2f}s")

    rng = np.random.default_rng(args.seed)
    coarse_res = args.coarse_res
    n_coarse = args.n_voxels
    if n_coarse > coarse_res**3:
        print(
            f"error: --n-voxels ({n_coarse}) cannot exceed coarse_res³ "
            f"({coarse_res}³ = {coarse_res**3})",
            file=sys.stderr,
        )
        return 2
    flat = rng.choice(coarse_res**3, size=n_coarse, replace=False)
    z = flat // (coarse_res**2)
    rem = flat % (coarse_res**2)
    y = rem // coarse_res
    x = rem % coarse_res
    coords = mx.array(np.stack([z, y, x], axis=-1).astype(np.int32))
    latent = mx.array(rng.standard_normal((n_coarse, 32)).astype(np.float32) * 0.5)

    print(f"running decoder ({n_coarse} latent voxels at {coarse_res}³) ...")
    t0 = time.perf_counter()
    out = decoder(latent, coords, coarse_resolution=coarse_res)
    mx.eval(out.coords, out.v, out.delta_logits, out.gamma)
    dec_t = time.perf_counter() - t0
    print(
        f"  {dec_t * 1000:.0f} ms  →  {out.coords.shape[0]} fine voxels at {out.output_resolution}³"
    )

    print("extracting mesh ...")
    t0 = time.perf_counter()
    verts, faces = extract_mesh(
        out.coords,
        out.v,
        out.delta_logits,
        out.gamma,
        grid_size=out.output_resolution,
    )
    ext_t = time.perf_counter() - t0
    print(f"  {ext_t * 1000:.0f} ms  →  {verts.shape[0]} vertices, {faces.shape[0]} triangles")

    if faces.shape[0] == 0:
        print(
            "warning: zero triangles — the random latent did not produce any "
            "active edges. Try a different --seed.",
            file=sys.stderr,
        )

    written = export_glb(verts, faces, args.output)
    print(f"wrote GLB: {written}  ({written.stat().st_size / 1024:.1f} KB)")
    print(
        "open in macOS Quick Look (spacebar in Finder), Blender, or "
        "https://gltf-viewer.donmccurdy.com/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
