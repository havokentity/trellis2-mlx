"""Mirror of upstream ``TextureMesh.json``.

NOT YET IMPLEMENTED in trellis2-mlx.

This script exists so that every example workflow has a corresponding
Python entry point with the right CLI parameters; running it today
reports the upstream nodes that we have not yet ported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path  # noqa: F401  (Path is used by some --mesh / --video args)
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples.workflows._common import add_common_args, not_implemented_stub


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='texture_mesh', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'TextureMesh.glb')
    parser.add_argument('--mesh', type=Path, default=None, help='External mesh (.obj or .glb)')
    parser.add_argument('--texture-resolution', type=int, default=4096, help='UV atlas size')
    return not_implemented_stub(
        parser, workflow_json='TextureMesh.json',
        missing_features=['Texture-on-arbitrary-mesh (texture pipeline today only works on our own SLAT-generated meshes, not arbitrary externally loaded geometry)', 'Trellis2LoadMesh node (loading external .obj/.glb as pipeline input)', 'UV-unwrap + 4096² texture atlas baking'],
        upstream_nodes=['Trellis2LoadMesh', 'Trellis2MeshTexturing(tex_res=4096)'],
        notes='For our SLAT-native texturing, use examples/workflows/low_poly.py or advanced.py.',
    )


if __name__ == "__main__":
    sys.exit(main())
