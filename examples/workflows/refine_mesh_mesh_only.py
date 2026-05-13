"""Mirror of upstream ``RefineMesh_MeshOnly.json``.

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
        prog='refine_mesh_mesh_only', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'RefineMesh_MeshOnly.glb')
    parser.add_argument('--mesh', type=Path, default=None, help='External mesh')
    parser.add_argument('--target-faces', type=int, default=500000, help='Decimation target')
    return not_implemented_stub(
        parser, workflow_json='RefineMesh_MeshOnly.json',
        missing_features=['MeshRefiner DiT model', 'Trellis2LoadMesh'],
        upstream_nodes=['Trellis2LoadMesh', 'Trellis2MeshRefiner'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
