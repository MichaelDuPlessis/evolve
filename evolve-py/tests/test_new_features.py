"""Tests for custom comparator, run_with collector, and parallel operators."""
import pytest


def make_ea(operators, termination, comparator=None, seed=42, population_size=20):
    from evolve_rs import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve_rs.initializers import RangedRandom

    def fitness(genome):
        return float(sum(genome))

    return EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=operators,
        fitness=fitness,
        termination=termination,
        population_size=population_size,
        comparator=comparator or Maximize(),
        seed=seed,
    )


# ── Feature 1: Custom Comparator ──────────────────────────────────────────────

def test_callable_comparator_maximize():
    """A callable comparator returning a > b behaves like Maximize."""
    from evolve_rs import EvolutionaryAlgorithm, MaxGenerations
    from evolve_rs.operators import Fill, RandomReset
    from evolve_rs.initializers import RangedRandom

    def maximize(a, b):
        return a > b

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=Fill(RandomReset()),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(10),
        population_size=20,
        comparator=maximize,
        seed=42,
    )
    result = ea.run()
    assert result.generations == 10
    # Best should be the highest fitness
    all_fitness = [ind.fitness for ind in result.population]
    assert result.best().fitness == max(all_fitness)


def test_callable_comparator_minimize():
    """A callable comparator returning a < b behaves like Minimize."""
    from evolve_rs import EvolutionaryAlgorithm, MaxGenerations
    from evolve_rs.operators import Fill, RandomReset
    from evolve_rs.initializers import RangedRandom

    def minimize(a, b):
        return a < b

    ea_min_callable = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=Fill(RandomReset()),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(10),
        population_size=20,
        comparator=minimize,
        seed=42,
    )
    result = ea_min_callable.run()
    assert result.generations == 10


def test_comparator_callable_is_called():
    """Verify callable comparator actually gets called."""
    from evolve_rs import EvolutionaryAlgorithm, MaxGenerations
    from evolve_rs.operators import Fill, RandomReset
    from evolve_rs.initializers import RangedRandom

    calls = []

    def tracking_comparator(a, b):
        calls.append((a, b))
        return a > b

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=Fill(RandomReset()),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(5),
        population_size=20,
        comparator=tracking_comparator,
        seed=42,
    )
    ea.run()
    # Should have been called at least once (in convert_result to find best)
    assert len(calls) > 0
    for a, b in calls:
        assert isinstance(a, float)
        assert isinstance(b, float)


def test_invalid_comparator_raises():
    """Passing a non-callable, non-Maximize/Minimize raises TypeError."""
    from evolve_rs import EvolutionaryAlgorithm, MaxGenerations
    from evolve_rs.operators import Fill, RandomReset
    from evolve_rs.initializers import RangedRandom

    with pytest.raises(TypeError):
        EvolutionaryAlgorithm(
            initializer=RangedRandom(10, 10),
            operators=Fill(RandomReset()),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(5),
            population_size=20,
            comparator="not a comparator",
        )


# ── Feature 2: Custom Collector (run_with) ─────────────────────────────────────

def test_run_with_basic_collector():
    """A collector with on_generation and finalize works."""
    from evolve_rs import MaxGenerations
    from evolve_rs.operators import Fill, RandomReset

    class HistoryCollector:
        def __init__(self):
            self.history = []

        def on_generation(self, generation, best_fitness):
            self.history.append((generation, best_fitness))

        def finalize(self):
            return self.history

    ea = make_ea(Fill(RandomReset()), MaxGenerations(5))
    result = ea.run_with(HistoryCollector())

    assert isinstance(result, list)
    assert len(result) == 5
    for i, (gen, fit) in enumerate(result):
        assert gen == i + 1
        assert isinstance(fit, float)


def test_run_with_on_start_called():
    """on_start hook is called once."""
    from evolve_rs import MaxGenerations
    from evolve_rs.operators import Fill, RandomReset

    class StartCollector:
        def __init__(self):
            self.start_calls = 0
            self.gen_calls = 0

        def on_start(self, generation, best_fitness):
            self.start_calls += 1

        def on_generation(self, generation, best_fitness):
            self.gen_calls += 1

        def finalize(self):
            return (self.start_calls, self.gen_calls)

    ea = make_ea(Fill(RandomReset()), MaxGenerations(3))
    start_calls, gen_calls = ea.run_with(StartCollector())

    assert start_calls == 1
    assert gen_calls == 3


def test_run_with_on_end_called():
    """on_end hook is called once."""
    from evolve_rs import MaxGenerations
    from evolve_rs.operators import Fill, RandomReset

    class EndCollector:
        def __init__(self):
            self.end_calls = 0

        def on_end(self, generation, best_fitness):
            self.end_calls += 1

        def finalize(self):
            return self.end_calls

    ea = make_ea(Fill(RandomReset()), MaxGenerations(3))
    end_calls = ea.run_with(EndCollector())

    assert end_calls == 1


def test_run_with_no_finalize_returns_none():
    """A collector without finalize returns None."""
    from evolve_rs import MaxGenerations
    from evolve_rs.operators import Fill, RandomReset

    class NoFinalizeCollector:
        def on_generation(self, generation, best_fitness):
            pass

    ea = make_ea(Fill(RandomReset()), MaxGenerations(3))
    result = ea.run_with(NoFinalizeCollector())

    assert result is None


def test_run_with_fitness_tracking():
    """Collector can track best fitness progression."""
    from evolve_rs import MaxGenerations
    from evolve_rs.operators import Fill, Pipeline, Tournament, RandomReset

    class FitnessTracker:
        def __init__(self):
            self.best_per_gen = []

        def on_generation(self, generation, best_fitness):
            self.best_per_gen.append(best_fitness)

        def finalize(self):
            return self.best_per_gen

    ea = make_ea(
        Fill(Pipeline([Tournament(3), RandomReset()])),
        MaxGenerations(10),
    )
    history = ea.run_with(FitnessTracker())

    assert len(history) == 10
    assert all(isinstance(f, float) for f in history)


def test_run_with_returns_finalize_value():
    """Whatever finalize() returns is the run_with return value."""
    from evolve_rs import MaxGenerations
    from evolve_rs.operators import Fill, RandomReset

    class DictCollector:
        def finalize(self):
            return {"done": True, "value": 42}

    ea = make_ea(Fill(RandomReset()), MaxGenerations(1))
    result = ea.run_with(DictCollector())

    assert result == {"done": True, "value": 42}


# ── Feature 3: Parallel Operators ─────────────────────────────────────────────

def test_parallel_random_reset_runs():
    """ParallelRandomReset can be used inside a ParallelFill."""
    from evolve_rs import MaxGenerations
    from evolve_rs.parallel import ParallelFill, ParallelRandomReset

    ea = make_ea(
        ParallelFill(ParallelRandomReset(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20


def test_parallel_swap_runs():
    """ParallelSwap works as inner operator."""
    from evolve_rs import MaxGenerations
    from evolve_rs.parallel import ParallelFill, ParallelSwap

    ea = make_ea(
        ParallelFill(ParallelSwap(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_inversion_runs():
    """ParallelInversion works."""
    from evolve_rs import MaxGenerations
    from evolve_rs.parallel import ParallelFill, ParallelInversion

    ea = make_ea(
        ParallelFill(ParallelInversion(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_scramble_runs():
    """ParallelScramble works."""
    from evolve_rs import MaxGenerations
    from evolve_rs.parallel import ParallelFill, ParallelScramble

    ea = make_ea(
        ParallelFill(ParallelScramble(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_creep_runs():
    """ParallelCreep works with integer genomes."""
    from evolve_rs import MaxGenerations
    from evolve_rs.parallel import ParallelFill, ParallelCreep

    ea = make_ea(
        ParallelFill(ParallelCreep(step=1), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_gaussian_runs():
    """ParallelGaussian works with float genomes."""
    from evolve_rs import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve_rs.parallel import ParallelFill, ParallelGaussian
    from evolve_rs.initializers import RangedRandom

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10, dtype="f64"),
        operators=ParallelFill(ParallelGaussian(std_dev=0.1), target_size=20),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(5),
        population_size=20,
        comparator=Maximize(),
        seed=42,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_import():
    """parallel module can be imported and has expected exports."""
    from evolve_rs import parallel

    assert hasattr(parallel, "ParallelFill")
    assert hasattr(parallel, "ParallelRandomReset")
    assert hasattr(parallel, "ParallelSwap")
    assert hasattr(parallel, "ParallelInversion")
    assert hasattr(parallel, "ParallelScramble")
    assert hasattr(parallel, "ParallelCreep")
    assert hasattr(parallel, "ParallelGaussian")


def test_parallel_creep_invalid_for_float():
    """ParallelCreep raises ValueError for float dtype."""
    from evolve_rs import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve_rs.parallel import ParallelFill, ParallelCreep
    from evolve_rs.initializers import RangedRandom

    with pytest.raises(ValueError, match="integer"):
        EvolutionaryAlgorithm(
            initializer=RangedRandom(10, 10, dtype="f64"),
            operators=ParallelFill(ParallelCreep(step=1), target_size=20),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(5),
            population_size=20,
            comparator=Maximize(),
        )
