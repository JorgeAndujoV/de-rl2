"""Suite-qualified benchmark registry.

Benchmarks are identified by '<suite>:<name>', e.g. 'cec13:f11'. This keeps
function 11 of CEC'13 distinct from function 11 of CEC'17 — different problems
that happen to share a number. Ids must be qualified; an unqualified id is an
error rather than a guess.

To add CEC'17 later: write benchmarks/cec17.py ending in a register_suite
call, then add one import at the bottom of this file. Nothing else changes.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BenchmarkSpec:
    """Everything needed to optimize and score one benchmark function."""

    id: str                  # 'cec13:f11'
    suite: str               # 'cec13'
    name: str                # 'Rastrigin'
    dim: int
    lower: float
    upper: float
    optimum: float           # f(x*), e.g. -400.0 for cec13:f11
    evaluate_np: Callable    # (D,)  -> float
    evaluate_tf: Callable    # (N,D) -> (N,)


_SUITE_BUILDERS = {}


def register_suite(suite, builder):
    """builder(name, dim) -> BenchmarkSpec"""
    if suite in _SUITE_BUILDERS:
        raise KeyError(f"Suite {suite!r} already registered.")
    _SUITE_BUILDERS[suite] = builder


def build_benchmark(benchmark_id, dim):
    """Instantiate a benchmark from its qualified id, e.g. 'cec13:f11'."""
    if ":" not in benchmark_id:
        raise KeyError(
            f"Benchmark ids must be suite-qualified, e.g. 'cec13:f11'. "
            f"Got {benchmark_id!r}."
        )
    suite, name = benchmark_id.split(":", 1)
    if suite not in _SUITE_BUILDERS:
        raise KeyError(
            f"Unknown suite {suite!r} in {benchmark_id!r}. "
            f"Available: {sorted(_SUITE_BUILDERS)}."
        )
    return _SUITE_BUILDERS[suite](name, dim)


# Imported for registration side effects; kept at the bottom so the functions
# above exist when the suite module runs.
from derl2.benchmarks import cec13  # noqa: E402,F401