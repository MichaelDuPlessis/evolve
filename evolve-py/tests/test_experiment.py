def test_experiment():
    from evolve_rs import Experiment, EvolutionaryAlgorithm, Maximize, MaxGenerations
    from evolve_rs.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
    from evolve_rs.initializers import RangedRandom

    def make_ea():
        return EvolutionaryAlgorithm(
            initializer=RangedRandom(20, 20),
            operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), SinglePoint(), RandomReset()])),
            fitness=lambda g: float(sum(g)),
            termination=MaxGenerations(10),
            population_size=50,
            comparator=Maximize(),
            seed=None,
        )

    results = Experiment(make_ea, trials=5).run()
    assert len(results) == 5
    for r in results:
        assert r.generations == 10
        assert r.best().fitness > 0
