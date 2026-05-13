"""Mirror of upstream ``RefineMesh.json``.

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
        prog='refine_mesh', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'RefineMesh.glb')
    parser.add_argument('--mesh', type=Path, default=None, help='External mesh to refine')
    parser.add_argument('--target-faces', type=int, default=500000, help='Decimation target')
    parser.add_argument('--texture-resolution', type=int, default=2048, help='UV atlas size')
    return not_implemented_stub(
        parser, workflow_json='RefineMesh.json',
        missing_features=['MeshRefiner DiT model', 'UV unwrap + rasterizer', 'Trellis2LoadMesh'],
        upstream_nodes=['Trellis2LoadMesh', 'Trellis2MeshRefiner', 'Trellis2UnWrapAndRasterizer'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
