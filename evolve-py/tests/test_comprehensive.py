"""Comprehensive tests for evolve-rs Python bindings."""
import pytest


def _make_simple_ea(**overrides):
    """Helper to create a basic EA with sensible defaults."""
    from evolve_rs import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve_rs.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
    from evolve_rs.initializers import RangedRandom

    defaults = dict(
        initializer=RangedRandom(20, 20),
        operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), SinglePoint(), RandomReset()])),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(10),
        population_size=50,
        comparator=Maximize(),
        seed=42,
    )
    defaults.update(overrides)
    return EvolutionaryAlgorithm(**defaults)


# --- Error handling ---

def test_population_size_zero():
    with pytest.raises(Exception):
        _make_simple_ea(population_size=0)


def test_tournament_size_zero():
    from evolve_rs.operators import Tournament, Fill
    with pytest.raises(BaseException):
        _make_simple_ea(operators=Fill(Tournament(0)))


def test_invalid_operator_type():
    with pytest.raises((TypeError, Exception)):
        _make_simple_ea(operators="not an operator")


def test_fitness_returning_non_float():
    ea = _make_simple_ea(fitness=lambda g: "not a float")
    with pytest.raises(BaseException):
        ea.run()


def test_fitness_raising_exception():
    def bad_fitness(g):
        raise ValueError("intentional error")
    ea = _make_simple_ea(fitness=bad_fitness)
    with pytest.raises(BaseException):
        ea.run()


# --- Operator construction ---

def test_fill_with_tournament_only():
    from evolve_rs.operators import Tournament, Fill
    ea = _make_simple_ea(operators=Fill(Tournament(3)))
    result = ea.run()
    assert result.generations == 10


def test_pipeline_crossover():
    from evolve_rs.operators import Tournament, SinglePoint, Combine, Pipeline, Fill
    ea = _make_simple_ea(operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), SinglePoint()])))
    result = ea.run()
    assert result.generations == 10


def test_pipeline_mutation_only():
    from evolve_rs.operators import Tournament, RandomReset, Pipeline, Fill
    ea = _make_simple_ea(operators=Fill(Pipeline([Tournament(3), RandomReset()])))
    result = ea.run()
    assert result.generations == 10


# --- Nesting combinators ---

def test_nested_pipeline_in_fill():
    from evolve_rs.operators import Tournament, SinglePoint, RandomReset, Combine, Pipeline, Fill
    # Tournament selects 1 individual; Combine([T, T]) provides 2 parents for SinglePoint
    ea = _make_simple_ea(operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), SinglePoint(), RandomReset()])))
    result = ea.run()
    assert result.generations == 10


def test_nested_combine_in_pipeline():
    from evolve_rs.operators import Tournament, SinglePoint, RandomReset, Combine, Pipeline, Fill
    ops = Fill(Pipeline([Combine([Tournament(3), Tournament(5)]), SinglePoint(), RandomReset()]))
    ea = _make_simple_ea(operators=ops)
    result = ea.run()
    assert result.generations == 10


def test_deeply_nested():
    from evolve_rs.operators import Tournament, SinglePoint, RandomReset, Combine, Pipeline, Fill
    ops = Fill(Pipeline([
        Combine([Tournament(3), Tournament(3)]),
        SinglePoint(),
        Pipeline([RandomReset()]),
    ]))
    ea = _make_simple_ea(operators=ops)
    result = ea.run()
    assert result.generations == 10


# --- Result properties ---

def test_result_properties():
    ea = _make_simple_ea()
    result = ea.run()

    assert result.total_duration > 0.0
    assert len(result.generation_durations) == result.generations
    assert len(result.best_fitness) == result.generations
    assert len(result.population) == 50

    for ind in result.population:
        assert len(ind.genome) == 20
        assert isinstance(ind.fitness, float)

    best = result.best()
    assert isinstance(best.genome, list)
    assert isinstance(best.fitness, float)
    assert best.fitness >= 0.0


# --- Edge cases ---

def test_single_generation():
    from evolve_rs import MaxGenerations
    ea = _make_simple_ea(termination=MaxGenerations(1))
    result = ea.run()
    assert result.generations == 1
    assert len(result.best_fitness) == 1


def test_population_size_one():
    ea = _make_simple_ea(population_size=1)
    result = ea.run()
    assert len(result.population) == 1


def test_short_genome():
    from evolve_rs.initializers import RangedRandom
    ea = _make_simple_ea(initializer=RangedRandom(1, 1))
    result = ea.run()
    best = result.best()
    assert len(best.genome) == 1
