"""Mirror of upstream ``ReconViaGen_MeshOnly_FromVideo.json``.

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
        prog='recon_via_gen_video', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser, 'ReconViaGen_MeshOnly_FromVideo.glb')
    parser.add_argument('--video', type=Path, default=None, help='Input video')
    parser.add_argument('--num-frames', type=int, default=8, help='Frames to extract')
    parser.add_argument('--target-faces', type=int, default=500000, help='Decimation target')
    return not_implemented_stub(
        parser, workflow_json='ReconViaGen_MeshOnly_FromVideo.json',
        missing_features=['ReconViaGen sparse generator', 'Video frame extraction (Trellis2ExtractImagesFromVideo)', 'Decomposed shape pipeline'],
        upstream_nodes=['Trellis2ExtractImagesFromVideo', 'Trellis2SparseGeneratorWithReconViaGen', 'Trellis2ShapeGenerator', 'Trellis2ShapeCascadeGenerator'],
        notes='',
    )


if __name__ == "__main__":
    sys.exit(main())
