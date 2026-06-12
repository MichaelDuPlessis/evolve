def test_ge_basic():
    from evolve_rs import EvolutionaryAlgorithm, Maximize, MaxGenerations, Grammar, StandardMapper, GeFitness
    from evolve_rs.operators import Tournament, SinglePoint, RandomReset, Pipeline, Combine, Fill
    from evolve_rs.initializers import RangedRandom

    grammar = Grammar.builder() \
        .rule("expr", [["expr", "op", "expr"], ["x"], ["1"]]) \
        .rule("op", [["+"], ["-"], ["*"]]) \
        .start("expr") \
        .build()

    ge_fitness = GeFitness(
        grammar=grammar,
        mapper=StandardMapper(max_wraps=3),
        evaluator=lambda phenotype: float(len(phenotype)),  # silly fitness
        penalty=0.0,
    )

    ea = EvolutionaryAlgorithm(
        initializer=RangedRandom(50, 200),
        operators=Fill(Pipeline([Combine([Tournament(3), Tournament(3)]), SinglePoint(), RandomReset()])),
        fitness=ge_fitness,
        termination=MaxGenerations(20),
        population_size=100,
        comparator=Maximize(),
        seed=42,
    )
    result = ea.run()
    assert result.generations == 20
    assert result.best().fitness > 0.0
