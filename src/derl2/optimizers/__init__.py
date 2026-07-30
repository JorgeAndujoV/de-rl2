"""Optimizer package.

The DE mutation strategies live in `strategies.py` (ported unchanged); the
STRATEGIES registry there IS the list of available strategies — to add one,
write the class and add one dictionary line. The optimizer's entire public
surface is a single function, `run_segment`, in `de.py`: one DE run over a
fixed FE budget, with no RL knowledge.

These names are re-exported here so callers use `derl2.optimizers` without
reaching into submodules.
"""

from derl2.optimizers.strategies import STRATEGIES, build_strategy
from derl2.optimizers.de import run_segment, SegmentResult

__all__ = ["STRATEGIES", "build_strategy", "run_segment", "SegmentResult"]
