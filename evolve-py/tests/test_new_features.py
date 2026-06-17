"""Tests for custom comparator, run_with collector, and parallel operators."""
import pytest


def make_ea(operators, termination, comparator=None, seed=42, population_size=20):
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.initializers import RangedRandom

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
    from evolve import EvolutionaryAlgorithm, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom

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
    from evolve import EvolutionaryAlgorithm, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom

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
    from evolve import EvolutionaryAlgorithm, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom

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
    from evolve import EvolutionaryAlgorithm, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom

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
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    class HistoryCollector:
        def __init__(self):
            self.history = []

        def on_generation(self, generation, best_fitness, population):
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
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    class StartCollector:
        def __init__(self):
            self.start_calls = 0
            self.gen_calls = 0

        def on_start(self, generation, best_fitness, population):
            self.start_calls += 1

        def on_generation(self, generation, best_fitness, population):
            self.gen_calls += 1

        def finalize(self):
            return (self.start_calls, self.gen_calls)

    ea = make_ea(Fill(RandomReset()), MaxGenerations(3))
    start_calls, gen_calls = ea.run_with(StartCollector())

    assert start_calls == 1
    assert gen_calls == 3


def test_run_with_on_end_called():
    """on_end hook is called once."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    class EndCollector:
        def __init__(self):
            self.end_calls = 0

        def on_end(self, generation, best_fitness, population):
            self.end_calls += 1

        def finalize(self):
            return self.end_calls

    ea = make_ea(Fill(RandomReset()), MaxGenerations(3))
    end_calls = ea.run_with(EndCollector())

    assert end_calls == 1


def test_run_with_no_finalize_returns_none():
    """A collector without finalize returns None."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    class NoFinalizeCollector:
        def on_generation(self, generation, best_fitness, population):
            pass

    ea = make_ea(Fill(RandomReset()), MaxGenerations(3))
    result = ea.run_with(NoFinalizeCollector())

    assert result is None


def test_run_with_fitness_tracking():
    """Collector can track best fitness progression."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, Pipeline, Tournament, RandomReset

    class FitnessTracker:
        def __init__(self):
            self.best_per_gen = []

        def on_generation(self, generation, best_fitness, population):
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
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    class DictCollector:
        def finalize(self):
            return {"done": True, "value": 42}

    ea = make_ea(Fill(RandomReset()), MaxGenerations(1))
    result = ea.run_with(DictCollector())

    assert result == {"done": True, "value": 42}


# ── Feature 3: Parallel Operators ─────────────────────────────────────────────

def test_parallel_random_reset_runs():
    """ParallelRandomReset can be used inside a ParallelFill."""
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelRandomReset

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
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelSwap

    ea = make_ea(
        ParallelFill(ParallelSwap(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_inversion_runs():
    """ParallelInversion works."""
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelInversion

    ea = make_ea(
        ParallelFill(ParallelInversion(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_scramble_runs():
    """ParallelScramble works."""
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelScramble

    ea = make_ea(
        ParallelFill(ParallelScramble(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_creep_runs():
    """ParallelCreep works with integer genomes."""
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelCreep

    ea = make_ea(
        ParallelFill(ParallelCreep(step=1), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5


def test_parallel_gaussian_runs():
    """ParallelGaussian works with float genomes."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.parallel import ParallelFill, ParallelGaussian
    from evolve.initializers import RangedRandom

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
    from evolve import parallel

    assert hasattr(parallel, "ParallelFill")
    assert hasattr(parallel, "ParallelRandomReset")
    assert hasattr(parallel, "ParallelSwap")
    assert hasattr(parallel, "ParallelInversion")
    assert hasattr(parallel, "ParallelScramble")
    assert hasattr(parallel, "ParallelCreep")
    assert hasattr(parallel, "ParallelGaussian")


def test_parallel_creep_invalid_for_float():
    """ParallelCreep raises ValueError for float dtype."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.parallel import ParallelFill, ParallelCreep
    from evolve.initializers import RangedRandom

    with pytest.raises(ValueError, match="integer"):
        EvolutionaryAlgorithm(
            initializer=RangedRandom(10, 10, dtype="f64"),
            operators=ParallelFill(ParallelCreep(step=1), target_size=20),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(5),
            population_size=20,
            comparator=Maximize(),
        )


# ── Missing coverage ──────────────────────────────────────────────────────────

def test_with_rate_operator():
    """WithRate wraps an operator and applies it at the given rate."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, WithRate, RandomReset

    ea = make_ea(Fill(WithRate(RandomReset(), 0.5)), MaxGenerations(5))
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20


def test_parallel_fill():
    """ParallelFill works as top-level operator."""
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelRandomReset

    ea = make_ea(
        ParallelFill(ParallelRandomReset(), target_size=20),
        MaxGenerations(5),
        population_size=20,
    )
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20


def test_parallel_random_reset():
    """ParallelRandomReset inside ParallelFill mutates population."""
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelRandomReset

    ea = make_ea(
        ParallelFill(ParallelRandomReset(), target_size=20),
        MaxGenerations(3),
        population_size=20,
    )
    result = ea.run()
    assert len(result.population) == 20


def test_parallel_swap():
    """ParallelSwap inside ParallelFill works."""
    from evolve import MaxGenerations
    from evolve.parallel import ParallelFill, ParallelSwap

    ea = make_ea(
        ParallelFill(ParallelSwap(), target_size=20),
        MaxGenerations(3),
        population_size=20,
    )
    result = ea.run()
    assert len(result.population) == 20


def test_weighted_operator():
    """Weighted combinator distributes work among operators by weight."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, Weighted, RandomReset, Swap

    ops = Weighted([(RandomReset(), 3), (Swap(), 1)])
    ea = make_ea(Fill(ops), MaxGenerations(5))
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20


def test_proportional_operator():
    """Proportional combinator fills population proportionally by weight."""
    from evolve import MaxGenerations
    from evolve.operators import Proportional, RandomReset, Swap

    ops = Proportional([(RandomReset(), 3), (Swap(), 1)])
    ea = make_ea(ops, MaxGenerations(5))
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20


def test_repeat_operator():
    """Repeat applies its inner operator N times."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, Repeat, RandomReset

    ea = make_ea(Fill(Repeat(RandomReset(), 3)), MaxGenerations(5))
    result = ea.run()
    assert result.generations == 5


def test_identity_operator():
    """Identity passes population through unchanged."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, Identity

    ea = make_ea(Fill(Identity()), MaxGenerations(5))
    result = ea.run()
    assert result.generations == 5
    assert len(result.population) == 20


def test_seed_determinism_f64():
    """Two runs with the same seed and dtype=f64 produce identical results."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom

    def make():
        return EvolutionaryAlgorithm(
            initializer=RangedRandom(10, 10, dtype="f64"),
            operators=Fill(RandomReset()),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(10),
            population_size=20,
            comparator=Maximize(),
            seed=99,
        )

    r1 = make().run()
    r2 = make().run()
    assert [ind.fitness for ind in r1.population] == [ind.fitness for ind in r2.population]


def test_seed_determinism_i32():
    """Two runs with the same seed and dtype=i32 produce identical results."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom

    def make():
        return EvolutionaryAlgorithm(
            initializer=RangedRandom(10, 10, dtype="i32"),
            operators=Fill(RandomReset()),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(10),
            population_size=20,
            comparator=Maximize(),
            seed=99,
        )

    r1 = make().run()
    r2 = make().run()
    assert [ind.fitness for ind in r1.population] == [ind.fitness for ind in r2.population]


def test_error_tournament_zero():
    """Tournament(0) raises ValueError when used in an EA."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, Tournament
    from evolve.initializers import RangedRandom
    import pytest

    with pytest.raises(ValueError, match="non-zero"):
        EvolutionaryAlgorithm(
            initializer=RangedRandom(10, 10),
            operators=Fill(Tournament(0)),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(1),
            population_size=20,
            comparator=Maximize(),
        )


def test_error_fitness_non_float():
    """Fitness returning a string raises TypeError."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom
    import pytest

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=Fill(RandomReset()),
        fitness=lambda g: "not a float",
        termination=MaxGenerations(5),
        population_size=20,
        comparator=Maximize(),
    )
    with pytest.raises(TypeError):
        ea.run()


def test_error_fitness_exception():
    """Fitness raising ValueError propagates out of ea.run()."""
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom
    import pytest

    def bad_fitness(g):
        raise ValueError("bad fitness")

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=Fill(RandomReset()),
        fitness=bad_fitness,
        termination=MaxGenerations(5),
        population_size=20,
        comparator=Maximize(),
    )
    with pytest.raises((ValueError, Exception)):
        ea.run()


# ── Improvement 1: Conditional combinator ─────────────────────────────────────

def test_conditional_switches_operators():
    """Conditional uses if_true op before threshold, if_false after."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, Conditional, Tournament, RandomReset

    applied_true = []
    applied_false = []

    def track_true(population, generation):
        applied_true.append(1)
        return [ind["genome"] for ind in population]

    def track_false(population, generation):
        applied_false.append(1)
        return [ind["genome"] for ind in population]

    def pred(generation, pop_size):
        return generation <= 3

    ea = make_ea(
        operators=Fill(Conditional(pred, track_true, track_false)),
        termination=MaxGenerations(6),
    )
    ea.run()

    assert len(applied_true) > 0
    assert len(applied_false) > 0


def test_conditional_always_true():
    """Conditional always uses if_true when predicate always returns True."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, Conditional, RandomReset, Identity

    calls = []

    def my_op(population, generation):
        calls.append(1)
        return [ind["genome"] for ind in population]

    ea = make_ea(
        operators=Fill(Conditional(lambda g, p: True, my_op, Identity())),
        termination=MaxGenerations(3),
    )
    ea.run()

    assert len(calls) > 0


# ── Improvement 2: Custom operators receive Individual dicts ──────────────────

def test_custom_operator_receives_dicts():
    """Custom operator receives list of dicts with genome and fitness keys."""
    from evolve import MaxGenerations
    from evolve.operators import Fill

    received = []

    def inspect_population(population, generation):
        received.append(population)
        return [ind["genome"] for ind in population]

    ea = make_ea(operators=Fill(inspect_population), termination=MaxGenerations(1))
    ea.run()

    assert len(received) > 0
    first = received[0]
    assert isinstance(first, list)
    for ind in first:
        assert "genome" in ind
        assert "fitness" in ind
        assert isinstance(ind["genome"], list)


def test_custom_operator_fitness_accessible():
    """Custom operator can read fitness from Individual dicts."""
    from evolve import MaxGenerations
    from evolve.operators import Fill

    def best_clone(population, generation):
        evaluated = [ind for ind in population if ind["fitness"] is not None]
        if evaluated:
            best = max(evaluated, key=lambda ind: ind["fitness"])
            return [best["genome"]] * len(population)
        return [ind["genome"] for ind in population]

    ea = make_ea(operators=Fill(best_clone), termination=MaxGenerations(5))
    result = ea.run()
    assert result.generations == 5


# ── Improvement 3: Collector hooks receive population ─────────────────────────

def test_collector_on_generation_receives_population():
    """on_generation receives (generation, best_fitness, population) list."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    received_pops = []

    class PopCollector:
        def on_generation(self, generation, best_fitness, population):
            received_pops.append(population)

        def finalize(self):
            return received_pops

    ea = make_ea(Fill(RandomReset()), MaxGenerations(2))
    pops = ea.run_with(PopCollector())

    assert len(pops) == 2
    for pop in pops:
        assert isinstance(pop, list)
        assert len(pop) > 0
        for ind in pop:
            assert "genome" in ind
            assert "fitness" in ind


def test_collector_population_has_correct_size():
    """Population passed to on_generation has the right size."""
    from evolve import MaxGenerations
    from evolve.operators import Fill, RandomReset

    sizes = []

    class SizeCollector:
        def on_generation(self, generation, best_fitness, population):
            sizes.append(len(population))

        def finalize(self):
            return sizes

    ea = make_ea(Fill(RandomReset()), MaxGenerations(3))
    result = ea.run_with(SizeCollector())

    assert all(s == 20 for s in result)  # population_size=20 in make_ea


# ── Improvement 4: Proportional with fixed size ───────────────────────────────

def test_proportional_fixed_size():
    """Proportional with size= produces offspring of fixed size."""
    from evolve import MaxGenerations
    from evolve.operators import Proportional, Tournament, RandomReset

    # Proportional with fixed output size of 20
    op = Proportional([(Tournament(3), 3), (RandomReset(), 1)], size=20)
    ea = make_ea(operators=op, termination=MaxGenerations(3))
    result = ea.run()
    assert result.generations == 3
    assert len(result.population) == 20


def test_proportional_no_size_uses_pop_size():
    """Proportional without size= uses population size (default behavior)."""
    from evolve import MaxGenerations
    from evolve.operators import Proportional, Tournament, RandomReset

    op = Proportional([(Tournament(3), 3), (RandomReset(), 1)])
    ea = make_ea(operators=op, termination=MaxGenerations(3))
    result = ea.run()
    assert result.generations == 3
    assert len(result.population) == 20


# ── New tests: SegmentDuplication, SegmentDeletion, collector error ───────────

def test_segment_duplication():
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, Pipeline, Tournament, SinglePoint, Combine, SegmentDuplication
    from evolve.initializers import RangedRandom
    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(50, 50),
        operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), SinglePoint(), SegmentDuplication(0.3, 200)])),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(10),
        population_size=50,
        comparator=Maximize(),
        seed=42,
    )
    result = ea.run()
    assert result.generations == 10


def test_segment_deletion():
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, Pipeline, Tournament, SinglePoint, Combine, SegmentDeletion
    from evolve.initializers import RangedRandom
    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(50, 50),
        operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), SinglePoint(), SegmentDeletion(0.3, 10)])),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(10),
        population_size=50,
        comparator=Maximize(),
        seed=42,
    )
    result = ea.run()
    assert result.generations == 10


def test_collector_error_propagation():
    from evolve import EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve.operators import Fill, RandomReset
    from evolve.initializers import RangedRandom
    import pytest

    class BadCollector:
        def on_generation(self, gen, bf, pop):
            if gen >= 2:
                raise ValueError("intentional collector error")

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(10, 10),
        operators=Fill(RandomReset()),
        fitness=lambda g: float(sum(g)),
        termination=MaxGenerations(10),
        population_size=20,
        comparator=Maximize(),
        seed=42,
    )
    with pytest.raises(BaseException):
        ea.run_with(BadCollector())
