"""Tests for the dtype system — all 10 genome types."""
import pytest


def _make_ea(dtype, *, initializer_class="ranged", genome_length=10, generations=5, pop_size=20):
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, RandomReset, Pipeline, Combine, Fill
    from evolve.initializers import RangedRandom, Random

    if initializer_class == "ranged":
        init = RangedRandom(genome_length, genome_length, dtype=dtype)
    else:
        init = Random(genome_length, dtype=dtype)

    ops = Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), RandomReset()]))

    return EvolutionaryAlgorithm(
        initializer=init,
        operators=ops,
        fitness=lambda g: float(sum(x if isinstance(x, (int, float)) else 0 for x in g)),
        termination=MaxGenerations(generations),
        population_size=pop_size,
        comparator=Maximize(),
        seed=42,
    )


@pytest.mark.parametrize("dtype", ["u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64"])
def test_dtype_ranged_random_runs(dtype):
    ea = _make_ea(dtype)
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20
    best = result.best()
    assert isinstance(best.fitness, float)
    assert isinstance(best.genome, list)
    assert len(best.genome) == 10


@pytest.mark.parametrize("dtype", ["u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64"])
def test_dtype_random_initializer_runs(dtype):
    ea = _make_ea(dtype, initializer_class="random")
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20


def test_float_genome_values_f64():
    """f64 genomes should produce list[float]."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, Gaussian, Fill
    from evolve.initializers import RangedRandom

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(5, 5, dtype="f64"),
        operators=Fill(Tournament(3)),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(3),
        population_size=10,
        comparator=Maximize(),
        seed=1,
    )
    result = ea.run()
    best = result.best()
    # f64 values should be floats
    assert all(isinstance(v, float) for v in best.genome)


def test_gaussian_mutation_f64():
    """Gaussian mutation should work with f64 dtype."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, Gaussian, Pipeline, Combine, Fill
    from evolve.initializers import RangedRandom

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10, dtype="f64"),
        operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), Gaussian(0.1)])),
        fitness=lambda g: -sum((x - 0.5) ** 2 for x in g),
        termination=MaxGenerations(5),
        population_size=20,
        comparator=Maximize(),
        seed=2,
    )
    result = ea.run()
    assert result.generations == 5


def test_gaussian_mutation_f32():
    """Gaussian mutation should work with f32 dtype."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, Gaussian, Pipeline, Combine, Fill
    from evolve.initializers import RangedRandom

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10, dtype="f32"),
        operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), Gaussian(0.1)])),
        fitness=lambda g: float(-sum((x - 0.5) ** 2 for x in g)),
        termination=MaxGenerations(5),
        population_size=20,
        comparator=Maximize(),
        seed=3,
    )
    result = ea.run()
    assert result.generations == 5


def test_arithmetic_crossover_f64():
    """Arithmetic crossover should work with f64 dtype."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, Arithmetic, Pipeline, Combine, Fill
    from evolve.initializers import RangedRandom

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10, dtype="f64"),
        operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), Arithmetic()])),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(5),
        population_size=20,
        comparator=Maximize(),
        seed=4,
    )
    result = ea.run()
    assert result.generations == 5


def test_creep_mutation_i32():
    """Creep mutation should work with i32 dtype."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, Creep, Pipeline, Fill
    from evolve.initializers import RangedRandom

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10, dtype="i32"),
        operators=Fill(Pipeline([Tournament(3), Creep(5)])),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(5),
        population_size=20,
        comparator=Maximize(),
        seed=5,
    )
    result = ea.run()
    assert result.generations == 5


def test_invalid_dtype_raises():
    from evolve.initializers import RangedRandom
    with pytest.raises(Exception, match="invalid dtype"):
        RangedRandom(10, 10, dtype="bad_dtype")


def test_gaussian_on_integer_dtype_raises():
    """Gaussian on integer dtype should raise ValueError."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, Gaussian, Fill
    from evolve.initializers import RangedRandom

    with pytest.raises(Exception):
        EvolutionaryAlgorithm(
            initializer=RangedRandom(5, 5, dtype="u8"),
            operators=Fill(Gaussian(0.1)),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(1),
            population_size=10,
            comparator=Maximize(),
            seed=6,
        )


def test_creep_on_float_dtype_raises():
    """Creep on float dtype should raise ValueError."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Tournament, Creep, Fill
    from evolve.initializers import RangedRandom

    with pytest.raises(Exception):
        EvolutionaryAlgorithm(
            initializer=RangedRandom(5, 5, dtype="f64"),
            operators=Fill(Creep(1)),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(1),
            population_size=10,
            comparator=Maximize(),
            seed=7,
        )
