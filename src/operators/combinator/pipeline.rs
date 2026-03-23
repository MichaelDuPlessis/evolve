use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};

/// Chains operators sequentially, feeding each operator's output as input to the next.
///
/// Accepts a tuple of up to 16 operators. The first operator receives the current
/// state's population; each subsequent operator receives the output of the previous one.
///
/// # Examples
///
/// ```
/// use evolve::operators::combinator::Pipeline;
/// use evolve::operators::crossover::SinglePoint;
/// use evolve::operators::mutation::RandomReset;
/// use evolve::operators::selection::TournamentSelection;
/// use std::num::NonZero;
///
/// // Select → Crossover → Mutate
/// let op = Pipeline::new((
///     TournamentSelection::new(NonZero::new(3).unwrap()),
///     SinglePoint::<u8>::new(),
///     RandomReset::<u8>::new(),
/// ));
/// ```
#[derive(Debug, Clone)]
pub struct Pipeline<O: ?Sized>(O);

impl<O> Pipeline<O> {
    /// Creates a new `Pipeline` from a tuple of operators.
    pub fn new(operators: O) -> Self {
        Self(operators)
    }
}

macro_rules! impl_genetic_pipeline {
    () => {};
    // Single operator (no chaining needed)
    ($first:ident) => {
        impl<G, F, Fe, R, C, $first> GeneticOperator<G, F, Fe, R, C>
            for Pipeline<($first,)>
        where
            $first: GeneticOperator<G, F, Fe, R, C>,
        {
            fn apply(
                &self,
                state: &State<G, F>,
                ctx: &mut Context<Fe, R, C>,
            ) -> Offspring<G, F> {
                self.0.0.apply(state, ctx)
            }
        }
    };
    // 2+ operators
    ($first:ident, $($rest:ident),+) => {
        impl<G, F, Fe, R, C, $first, $($rest),*> GeneticOperator<G, F, Fe, R, C>
            for Pipeline<($first, $($rest,)*)>
        where
            $first: GeneticOperator<G, F, Fe, R, C>,
            $($rest: GeneticOperator<G, F, Fe, R, C>),*,
        {
            fn apply(
                &self,
                state: &State<G, F>,
                ctx: &mut Context<Fe, R, C>,
            ) -> Offspring<G, F> {
                #[allow(non_snake_case)]
                let ($first, $($rest,)*) = &self.0;

                // Step 1: apply first operator
                let mut current_population = match $first.apply(state, ctx) {
                    Offspring::Single(ind) => {
                        let mut p = Population::new();
                        p.add(ind);
                        p
                    }
                    Offspring::Multiple(p) => p,
                };

                // Step 2: apply remaining operators
                $(
                    let next_state = state.with_population(current_population);

                    current_population = match $rest.apply(&next_state, ctx) {
                        Offspring::Single(ind) => {
                            let mut p = Population::new();
                            p.add(ind);
                            p
                        }
                        Offspring::Multiple(p) => p,
                    };
                )*

                Offspring::Multiple(current_population)
            }
        }

        impl_genetic_pipeline!($($rest),*);
    };
}

impl_genetic_pipeline!(
    T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16
);

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Pipeline<[O]>
where
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        // this is bad, assuming its safe to just index
        let mut offspring = self.0[0].apply(state, ctx);

        for operator in &self.0[1..] {
            let population = offspring.into_population();
            let state = state.with_population(population);
            offspring = operator.apply(&state, ctx);
        }

        offspring
    }
}
