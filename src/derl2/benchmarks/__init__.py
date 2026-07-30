"""Benchmark suites.

Benchmarks are addressed by suite-qualified id — 'cec13:f11' — and built with
build_benchmark(id, dim). See registry.py for how to add a suite.
"""

from derl2.benchmarks.registry import BenchmarkSpec, build_benchmark

__all__ = ["BenchmarkSpec", "build_benchmark"]