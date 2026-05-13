"""Mirror of upstream ``MeshRefiner.json``.

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
        prog='mesh_refiner', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'MeshRefiner.glb')
    parser.add_argument('--mesh', type=Path, default=None, help='External mesh to refine')
    parser.add_argument('--refiner-steps', type=int, default=25, help='Refiner sampler steps')
    parser.add_argument('--refiner-cfg', type=float, default=3.0, help='Refiner CFG')
    return not_implemented_stub(
        parser, workflow_json='MeshRefiner.json',
        missing_features=['MeshRefiner DiT model (second-pass image-conditioned geometry refinement)', 'Trellis2LoadMesh (load external mesh as input)'],
        upstream_nodes=['Trellis2LoadMesh', 'Trellis2MeshRefiner', 'Trellis2Remesh', 'Trellis2SimplifyMesh'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
