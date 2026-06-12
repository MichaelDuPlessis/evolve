"""Integration test: OneMax problem (maximize sum of genome bytes)."""
import pytest


def test_onemax_basic():
    """Run OneMax and verify the EA produces improving fitness."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
    from evolve.initializers import RangedRandom

    def fitness(genome):
        return float(sum(genome))

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(100, 100),
        operators=Fill(Pipeline([
            Combine([Tournament(3), Tournament(3)]),
            SinglePoint(),
            RandomReset(),
        ])),
        fitness=fitness,
        termination=MaxGenerations(100),
        population_size=200,
        comparator=Maximize(),
    )

    result = ea.run()

    # Should have run 100 generations
    assert result.generations == 100

    # Best fitness should be improving (later generations >= earlier)
    assert result.best_fitness[-1] >= result.best_fitness[0]

    # Best individual should have a good fitness
    best = result.best()
    assert best.fitness > 0
    assert len(best.genome) == 100

    # Population should have the right size
    assert len(result.population) == 200


def test_onemax_seeded_determinism():
    """Two runs with the same seed should produce identical results."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
    from evolve.initializers import RangedRandom

    def fitness(genome):
        return float(sum(genome))

    def make_ea(seed):
        return EvolutionaryAlgorithm(
            initializer=RangedRandom(50, 50),
            operators=Fill(Pipeline([
                Combine([Tournament(3), Tournament(3)]),
                SinglePoint(),
                RandomReset(),
            ])),
            fitness=fitness,
            termination=MaxGenerations(20),
            population_size=50,
            comparator=Maximize(),
            seed=seed,
        )

    ea1 = make_ea(42)
    ea2 = make_ea(42)

    result1 = ea1.run()
    result2 = ea2.run()

    assert result1.best_fitness == result2.best_fitness
    assert result1.best().genome == result2.best().genome


def test_onemax_minimize():
    """Verify Minimize comparator works (minimizes fitness)."""
    from evolve import EvolutionaryAlgorithm, Minimize, MaxGenerations
    from evolve.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
    from evolve.initializers import RangedRandom

    def fitness(genome):
        return float(sum(genome))

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(50, 50),
        operators=Fill(Pipeline([
            Combine([Tournament(3), Tournament(3)]),
            SinglePoint(),
            RandomReset(),
        ])),
        fitness=fitness,
        termination=MaxGenerations(50),
        population_size=100,
        comparator=Minimize(),
        seed=123,
    )

    result = ea.run()

    # Best fitness should be decreasing for minimize
    assert result.best_fitness[-1] <= result.best_fitness[0]
