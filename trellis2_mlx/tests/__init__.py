"""Test suite.

Conventions (per ``CLAUDE.md``):

* ``@pytest.mark.metal``    — requires a Metal-capable GPU.
* ``@pytest.mark.reference`` — compares MLX output against PyTorch CPU reference.
* ``@pytest.mark.slow``     — runs longer than ~5s.
"""

from __future__ import annotations
