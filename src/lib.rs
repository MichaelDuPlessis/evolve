use std::{marker::PhantomData, num::NonZero};

/// Individual with cached fitness
pub struct Individual<G, F> {
    pub genome: G,
    pub fitness: Option<F>,
}

impl<G, F> Individual<G, F> {
    /// Create a new `Indivdual` from a genome.
    pub fn new(genome: G) -> Self {
        Self {
            genome,
            fitness: None,
        }
    }

    /// Given a `FitnessEvaluation` calculate the `Individuals` fitness.
    pub fn calculate_fitness(&mut self, fitness_evaluation: &impl FitnessEvaluation<G, F>) {
        self.fitness = Some(fitness_evaluation.evaluate(&self.genome))
    }
}

/// The population in the algorithm. It is a list of Individuals.
pub struct Population<G, F> {
    individuals: Vec<Individual<G, F>>,
}

impl<G, F> Population<G, F> {
    /// Create a new population from a `Vec` of `Individuals`.
    pub fn from_individuals(individuals: Vec<Individual<G, F>>) -> Self {
        Self { individuals }
    }

    /// Create a new population from a `Vec` of genomes.
    pub fn from_genomes(genomes: Vec<G>) -> Self {
        let individuals: Vec<Individual<G, F>> = genomes.into_iter().map(Individual::new).collect();
        Self::from_individuals(individuals)
    }

    pub fn individuals(&self) -> &[Individual<G, F>] {
        &self.individuals
    }

    pub fn individuals_mut(&mut self) -> &mut [Individual<G, F>] {
        &mut self.individuals
    }

    pub fn into_vec(self) -> Vec<Individual<G, F>> {
        self.individuals
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Get the best individual in the Population
    pub fn best(self) -> Individual<G, F>
    where
        F: PartialOrd,
    {
        self.individuals
            .into_iter()
            .max_by(|a, b| {
                a.fitness
                    .as_ref()
                    .unwrap()
                    .partial_cmp(b.fitness.as_ref().unwrap())
                    .unwrap()
            })
            .expect("population cannot be empty")
    }
}

impl<G, F> From<Vec<Individual<G, F>>> for Population<G, F> {
    fn from(value: Vec<Individual<G, F>>) -> Self {
        Self::from_individuals(value)
    }
}

impl<G, F> From<Vec<G>> for Population<G, F> {
    fn from(value: Vec<G>) -> Self {
        Self::from_genomes(value)
    }
}

/// Trait to evaluate fitness
pub trait FitnessEvaluation<G, F> {
    fn evaluate(&self, genome: &G) -> F;
}

/// Trait to initialize the population
pub trait Initialization<G> {
    fn initialize(&self, population_size: NonZero<usize>) -> Vec<G>;
}

/// Termination condition
pub trait TerminationCondition {
    fn should_terminate(&self, generation: usize) -> bool;
}

/// Genetic operator trait — owns input population, returns a new population
pub trait GeneticOperator<G, F> {
    fn apply(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F>;
}

/// Trait to apply a pipeline of operators
pub trait ApplyOperators<G, F> {
    fn apply_operators(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F>;
}

/// Base case: single operator
impl<G, F, O> ApplyOperators<G, F> for (O,)
where
    O: GeneticOperator<G, F>,
{
    fn apply_operators(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F> {
        self.0.apply(population, fitness)
    }
}

/// Recursive case: operator + rest
impl<G, F, O, Rest> ApplyOperators<G, F> for (O, Rest)
where
    O: GeneticOperator<G, F>,
    Rest: ApplyOperators<G, F>,
{
    fn apply_operators(
        &self,
        population: Population<G, F>,
        fitness: &impl FitnessEvaluation<G, F>,
    ) -> Population<G, F> {
        let intermediate = self.0.apply(population, fitness);
        self.1.apply_operators(intermediate, fitness)
    }
}

/// Genetic Algorithm runner
pub struct GeneticAlgorithm<G, F, I, T, Fe, Ops>
where
    F: PartialOrd,
    I: Initialization<G>,
    T: TerminationCondition,
    Fe: FitnessEvaluation<G, F>,
    Ops: ApplyOperators<G, F>,
{
    initialization: I,
    termination: T,
    fitness: Fe,
    operators: Ops,
    population_size: NonZero<usize>,
    _marker: PhantomData<(G, F)>,
}

impl<G, F, I, T, Fe, Ops> GeneticAlgorithm<G, F, I, T, Fe, Ops>
where
    F: PartialOrd,
    I: Initialization<G>,
    T: TerminationCondition,
    Fe: FitnessEvaluation<G, F>,
    Ops: ApplyOperators<G, F>,
{
    pub fn new(
        initialization: I,
        termination: T,
        fitness: Fe,
        operators: Ops,
        population_size: NonZero<usize>,
    ) -> Self {
        Self {
            initialization,
            termination,
            fitness,
            operators,
            population_size,
            _marker: PhantomData,
        }
    }

    pub fn run(&self) -> Individual<G, F> {
        let genomes = self.initialization.initialize(self.population_size);
        let mut population = Population::from(genomes);

        let mut generation = 0;
        while !self.termination.should_terminate(generation) {
            // Apply pipeline — ownership flows through
            population = self.operators.apply_operators(population, &self.fitness);
            generation += 1;
        }

        population.best()
    }
}
