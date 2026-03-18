use crate::{
    core::{
        context::{Context, State},
        offspring::Offpring,
        population::Population,
    },
    fitness::FitnessEvaluator,
    operators::GeneticOperator,
};
use rand::Rng;

/// Combines multiple `GeneticOperators` and merges their ouputs.
pub struct Combine<O>(O);

impl<O> Combine<O> {
    pub fn new(operators: O) -> Self {
        Self(operators)
    }
}

macro_rules! impl_genetic_combine {
    // Base case: do nothing
    () => {};
    // Special case for a single element tuple: Return the result directly
    ($first:ident) => {
        impl<G, F, Fe, R, C, $first> GeneticOperator<G, F, Fe, R, C> for Combine<($first,)>
        where
            $first: GeneticOperator<G, F, Fe, R, C>,
            Fe: FitnessEvaluator<G, F>,
            R: Rng,
        {
            fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offpring<G, F> {
                self.0.0.apply(state, ctx)
            }
        }
    };
    // Recursive case for 2+ elements
    ($first:ident, $($rest:ident),+) => {
        impl<G, F, Fe, R, C, $first, $($rest),*> GeneticOperator<G, F, Fe, R, C> for Combine<($first, $($rest,)*)>
        where
            $first: GeneticOperator<G, F, Fe, R, C>,
            $($rest: GeneticOperator<G, F, Fe, R, C>),*,
            Fe: FitnessEvaluator<G, F>,
            R: Rng,
        {
            fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offpring<G, F> {
                #[allow(non_snake_case)]
                let ($first, $($rest,)*) = &self.0;

                let mut population = Population::new();

                // Apply the first operator
                population.add_offspring($first.apply(state, ctx));

                // Apply all subsequent operators in the tuple
                $(
                    population.add_offspring($rest.apply(state, ctx));
                )*

                Offpring::Multiple(population)
            }
        }

        impl_genetic_combine!($($rest),*);
    };
}

impl_genetic_combine!(
    T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16
);

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Combine<&[O]>
where
    O: GeneticOperator<G, F, Fe, R, C>,
    Fe: FitnessEvaluator<G, F>,
    R: Rng,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offpring<G, F> {
        let mut population = Population::new();

        for operator in self.0 {
            population.add_offspring(operator.apply(state, ctx));
        }

        Offpring::Multiple(population)
    }
}
