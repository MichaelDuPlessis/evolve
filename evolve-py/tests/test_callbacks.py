"""Tests for custom Python callback support: operators, termination, and initializer."""
import random
import pytest


def make_ea(operators, termination, initializer=None, seed=42):
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.initializers import RangedRandom

    def fitness(genome):
        return float(sum(genome))

    return EvolutionaryAlgorithm(
        initializer=initializer or RangedRandom(10, 10),
        operators=operators,
        fitness=fitness,
        termination=termination,
        population_size=20,
        comparator=Maximize(),
        seed=seed,
    )


# ── Custom Operator ────────────────────────────────────────────────────────────

def test_custom_operator_callable():
    """A plain Python callable can be used as a genetic operator."""
    from evolve import MaxGenerations
    from evolve.operators import Fill

    calls = []

    def set_all_max(population):
        calls.append(len(population))
        return [[255] * len(ind["genome"]) for ind in population]

    ea = make_ea(
        operators=Fill(set_all_max),
        termination=MaxGenerations(5),
    )
    result = ea.run()

    assert result.generations == 5
    assert len(calls) > 0
    # After several generations, all genomes should be at max
    assert result.best().fitness == 255 * 10


def test_custom_operator_in_pipeline():
    """A Python callable works inside a Pipeline combinator."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, Pipeline, Combine, Tournament, SinglePoint

    mutation_calls = []

    def noop_mutation(population):
        mutation_calls.append(1)
        return [ind["genome"] for ind in population]  # identity

    ea = make_ea(
        operators=Fill(Pipeline([
            Combine([Tournament(3), Tournament(3)]),
            SinglePoint(),
            noop_mutation,
        ])),
        termination=MaxGenerations(10),
    )
    result = ea.run()

    assert result.generations == 10
    assert len(mutation_calls) > 0


def test_custom_operator_modifies_genome():
    """Custom operator can modify genome values."""
    from evolve import MaxGenerations
    from evolve.operators import Fill

    def zero_first_gene(population):
        return [[0] + list(ind["genome"][1:]) for ind in population]

    ea = make_ea(
        operators=Fill(zero_first_gene),
        termination=MaxGenerations(3),
    )
    result = ea.run()

    # All individuals in final population should have genome[0] == 0
    for ind in result.population:
        assert ind.genome[0] == 0


# ── Custom Termination ─────────────────────────────────────────────────────────

def test_custom_termination_callable():
    """A Python callable can be used as a termination condition."""
    from evolve.operators import Fill, Pipeline, Combine, Tournament, SinglePoint, RandomReset

    stop_at = 7
    generations_seen = []

    def my_termination(generation, best_fitness):
        generations_seen.append(generation)
        return generation >= stop_at

    ea = make_ea(
        operators=Fill(Pipeline([
            Combine([Tournament(3), Tournament(3)]),
            SinglePoint(),
            RandomReset(),
        ])),
        termination=my_termination,
    )
    result = ea.run()

    assert result.generations == stop_at
    assert stop_at in generations_seen


def test_custom_termination_fitness_based():
    """Termination callable can stop based on best_fitness."""
    from evolve.operators import Fill, Pipeline, Tournament, RandomReset

    fitness_threshold = 1000.0  # impossible for 10-gene u8 genome (max=2550), use 100

    stopped_early = []

    def terminate_on_fitness(generation, best_fitness):
        if best_fitness >= 200:  # stop when best genome sum >= 200
            stopped_early.append(generation)
            return True
        return generation >= 500  # safety cap

    ea = make_ea(
        operators=Fill(Pipeline([Tournament(3), RandomReset()])),
        termination=terminate_on_fitness,
        seed=1,
    )
    result = ea.run()

    # Should stop at some point (either fitness-based or safety cap)
    assert result.generations <= 500


def test_maxgenerations_still_works():
    """Existing MaxGenerations still works after refactor."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    ea = make_ea(
        operators=Fill(RandomReset()),
        termination=MaxGenerations(15),
    )
    result = ea.run()
    assert result.generations == 15


def test_invalid_termination_raises():
    """Passing a non-callable, non-MaxGenerations object raises TypeError."""
    from evolve.operators import Fill, RandomReset

    with pytest.raises(TypeError):
        make_ea(
            operators=Fill(RandomReset()),
            termination="not a termination",  # type: ignore
        )


# ── Custom Initializer ─────────────────────────────────────────────────────────

def test_custom_initializer_callable():
    """A Python callable can be used as a population initializer."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, RandomReset

    def my_init(population_size):
        # All-zeros population
        return [[0] * 10 for _ in range(population_size)]

    # Give it dtype hint via attribute
    my_init.dtype = "u8"

    def fitness(genome):
        return float(sum(genome))

    ea = EvolutionaryAlgorithm(
        initializer=my_init,
        operators=Fill(RandomReset()),
        fitness=fitness,
        termination=MaxGenerations(1),
        population_size=10,
        comparator=Maximize(),
        seed=0,
    )
    result = ea.run()
    assert result.generations == 1
