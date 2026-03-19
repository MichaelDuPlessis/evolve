use crate::{
    core::{
        context::{Context, State},
        offspring::Offspring,
        population::Population,
    },
    operators::GeneticOperator,
};
use rand::{Rng, RngExt};
use std::num::NonZero;

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
        {
            fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
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
        {
            fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
                #[allow(non_snake_case)]
                let ($first, $($rest,)*) = &self.0;

                let mut population = Population::new();

                // Apply the first operator
                population.add_offspring($first.apply(state, ctx));

                // Apply all subsequent operators in the tuple
                $(
                    population.add_offspring($rest.apply(state, ctx));
                )*

                Offspring::Multiple(population)
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
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let mut population = Population::new();

        for operator in self.0 {
            population.add_offspring(operator.apply(state, ctx));
        }

        Offspring::Multiple(population)
    }
}

/// Repeat the `GeneticOperator` provided n times.
pub struct Repeat<O> {
    operator: O,
    n: usize,
}

impl<O> Repeat<O> {
    pub fn new(operator: O, n: usize) -> Self {
        Self { operator, n }
    }
}

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Repeat<O>
where
    O: GeneticOperator<G, F, Fe, R, C>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let mut population = Population::new();
        for _ in 0..self.n {
            population.add_offspring(self.operator.apply(state, ctx));
        }

        Offspring::Multiple(population)
    }
}

/// Weighted operator over a tuple of (Operator, weight)
pub struct Weighted<O>(O);

impl<O> Weighted<O> {
    pub fn new(operators: O) -> Self {
        Self(operators)
    }
}

macro_rules! impl_weighted_genetic {
    // Base case
    () => {};
    // Recursive case
    ($($Op:ident, $W:ident),+) => {
        impl<G, F, Fe, R, C, $($Op),*> GeneticOperator<G, F, Fe, R, C> for Weighted<( $(($Op, NonZero<u16>),)* )>
        where
            $($Op: GeneticOperator<G, F, Fe, R, C>),*,
            R: Rng,
        {
            #[allow(unused_assignments)]
            fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
                #[allow(non_snake_case)]
                let ( $( ($Op, $W), )* ) = &self.0;

                let total_weight: u16 = 0 $( + $W.get() )*;

                let mut roll = ctx.rng().random_range(0..total_weight);

                $(
                    let weight = $W.get();
                    if roll < weight {
                        return $Op.apply(state, ctx);
                    }
                    roll -= weight;
                )*

                // Fallback for rounding/empty cases (shouldn't be reached if total > 0)
                unreachable!("Weighted selection failed")
            }
        }

        // Helper to peel off the last pair for recursion
        impl_weighted_genetic_recurse!($($Op, $W),+);
    };
}

macro_rules! impl_weighted_genetic_recurse {
    ($OpHead:ident, $WHead:ident, $($OpRest:ident, $WRest:ident),+) => {
        impl_weighted_genetic!($($OpRest, $WRest),+);
    };
    ($OpHead:ident, $WHead:ident) => {};
}

impl_weighted_genetic!(
    O1, W1, O2, W2, O3, W3, O4, W4, O5, W5, O6, W6, O7, W7, O8, W8, O9, W9, O10, W10, O11, W11,
    O12, W12, O13, W13, O14, W14, O15, W15, O16, W16
);

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Weighted<&[(O, NonZero<u16>)]>
where
    O: GeneticOperator<G, F, Fe, R, C>,
    R: Rng,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let total_weight: u16 = self.0.iter().map(|(_, w)| w.get()).sum();

        let mut roll = ctx.rng().random_range(0..total_weight);

        for (operator, weight) in self.0 {
            let weight = weight.get();
            if roll < weight {
                return operator.apply(state, ctx);
            }
            roll -= weight;
        }

        unreachable!("Weighted selection failed")
    }
}

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

struct FixedSize(usize);

impl<G, F> GetSize<G, F> for FixedSize {
    fn get_size(&self, _: &State<G, F>) -> usize {
        self.0
    }
}

struct PopSize;

impl<G, F> GetSize<G, F> for PopSize {
    fn get_size(&self, state: &State<G, F>) -> usize {
        state.population().len()
    }
}

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
            size: PopSize,
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
