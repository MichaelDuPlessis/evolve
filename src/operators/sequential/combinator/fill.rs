use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};

/// Determines the target size for a [`Fill`] operator.
pub trait GetSize<G, F> {
    /// Returns the target population size.
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

#[derive(Debug, Clone, Copy)]
pub struct FixedSize(usize);

impl<G, F> GetSize<G, F> for FixedSize {
    fn get_size(&self, _: &State<G, F>) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PopSize(());

impl<G, F> GetSize<G, F> for PopSize {
    fn get_size(&self, state: &State<G, F>) -> usize {
        state.population().len()
    }
}

/// Repeats an operator until the output reaches a target population size.
///
/// Useful for ensuring the next generation has the right number of individuals,
/// regardless of how many each operator invocation produces. If the last batch
/// would overshoot the target, it is truncated.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::combinator::Fill;
/// use evolve::operators::sequential::mutation::RandomReset;
///
/// // Maintain the same population size each generation
/// let op = Fill::from_population_size(RandomReset::<u8>::new());
///
/// // Or fill to a specific size
/// let op = Fill::from_fixed_size(RandomReset::<u8>::new(), 200);
/// ```
#[derive(Debug, Clone)]
pub struct Fill<O, S> {
    operator: O,
    size: S,
}

impl<O, S> Fill<O, S> {
    /// Creates a new `Fill` with a custom size strategy.
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
