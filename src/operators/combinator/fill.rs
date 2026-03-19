use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};

/// Helper trait for the `Fill` operator.
pub trait GetSize<G, F> {
    fn get_size(&self, state: &State<G, F>) -> usize;
}

impl<G, F, T> GetSize<G, F> for T
where
    T: Fn(&State<G, F>) -> usize,
{
    fn get_size(&self, state: &State<G, F>) -> usize {
        (self)(state)
    }
}

#[derive(Debug)]
pub struct FixedSize(usize);

impl<G, F> GetSize<G, F> for FixedSize {
    fn get_size(&self, _: &State<G, F>) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub struct PopSize(());

impl<G, F> GetSize<G, F> for PopSize {
    fn get_size(&self, state: &State<G, F>) -> usize {
        state.population().len()
    }
}

#[derive(Debug)]
pub struct Fill<O, S> {
    operator: O,
    size: S,
}

impl<O, S> Fill<O, S> {
    pub fn new(operator: O, size: S) -> Self {
        Self { operator, size }
    }
}

impl<O> Fill<O, PopSize> {
    /// Creates a new `Fill` that maintains the input population size.
    pub fn from_population_size(operator: O) -> Self {
        Self {
            operator,
            size: PopSize(()),
        }
    }
}

impl<O> Fill<O, FixedSize> {
    /// Creates a new `Fill` that applies the underlying operator until a fixed size is met.
    pub fn from_fixed_size(operator: O, size: usize) -> Self {
        Self {
            operator,
            size: FixedSize(size),
        }
    }
}

impl<G, F, Fe, R, C, O, S> GeneticOperator<G, F, Fe, R, C> for Fill<O, S>
where
    O: GeneticOperator<G, F, Fe, R, C>,
    S: GetSize<G, F>, // TODO: see if this can instead impl FnOnce
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let target_size = self.size.get_size(state);

        let mut population = Population::with_capacity(target_size);

        while population.len() < target_size {
            let offspring = self.operator.apply(state, ctx);

            match offspring {
                Offspring::Single(ind) => population.add(ind),
                Offspring::Multiple(mut p) => {
                    // this is just to ensure no more than what is needed is added
                    let space_left = target_size - population.len();
                    if p.len() > space_left {
                        p.cull(space_left);
                    }
                    population.extend(p);
                }
            }
        }

        Offspring::Multiple(population)
    }
}
