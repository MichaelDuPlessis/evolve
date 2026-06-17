"""Parallel genetic operators for evolve-rs.

Parallel operators distribute work across available CPU cores.
They require a fixed ``target_size`` for Fill since population size
cannot be inferred at construction time.

Usage::

    from evolve.parallel import ParallelFill, ParallelRandomReset
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.initializers import RangedRandom

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=ParallelFill(ParallelRandomReset(), target_size=100),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(50),
        population_size=100,
    )
"""

from evolve._evolve import (
    ParallelFill,
    ParallelRandomReset,
    ParallelSwap,
    ParallelInversion,
    ParallelScramble,
    ParallelCreep,
    ParallelGaussian,
)

__all__ = [
    "ParallelFill",
    "ParallelRandomReset",
    "ParallelSwap",
    "ParallelInversion",
    "ParallelScramble",
    "ParallelCreep",
    "ParallelGaussian",
]
