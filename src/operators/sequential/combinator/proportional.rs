use crate::{
    core::{context::Context, offspring::Offspring, population::Population, state::State},
    operators::GeneticOperator,
};
use std::num::NonZero;

use super::fill::{Fill, FixedSize, GetSize, PopSize};

/// Splits the output population proportionally across multiple operators.
///
/// Each operator fills a portion of the output population based on its weight.
/// By default the output size matches the input population size, but a fixed
/// size can be specified via [`Proportional::with_size`].
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::combinator::Proportional;
/// use evolve::operators::sequential::mutation::RandomReset;
/// use std::num::NonZero;
///
/// // 2/3 from first operator, 1/3 from second (output matches input size)
/// let ops = vec![
///     (RandomReset::<u8>::new(), NonZero::new(2u16).unwrap()),
///     (RandomReset::<u8>::new(), NonZero::new(1u16).unwrap()),
/// ];
/// let op = Proportional::new(&ops);
///
/// // Fixed output size of 100
/// let ops = vec![
///     (RandomReset::<u8>::new(), NonZero::new(2u16).unwrap()),
///     (RandomReset::<u8>::new(), NonZero::new(1u16).unwrap()),
/// ];
/// let op = Proportional::with_size(&ops, 100);
/// ```
#[derive(Debug, Clone)]
pub struct Proportional<O, S = PopSize> {
    operators: O,
    size: S,
}

impl<O> Proportional<O> {
    /// Creates a new `Proportional` that outputs the same size as the input population.
    pub fn new(operators: O) -> Self {
        Self {
            operators,
            size: PopSize::new(),
        }
    }
}

impl<O> Proportional<O, FixedSize> {
    /// Creates a new `Proportional` with a fixed output size.
    pub fn with_size(operators: O, size: usize) -> Self {
        Self {
            operators,
            size: FixedSize::new(size),
        }
    }
}

impl<G, F, Fe, R, C, O, S> GeneticOperator<G, F, Fe, R, C>
    for Proportional<&[(O, NonZero<u16>)], S>
where
    O: GeneticOperator<G, F, Fe, R, C>,
    S: GetSize<G, F>,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let target_size = self.size.get_size(state);
        let total_weight: u16 = self.operators.iter().map(|(_, w)| w.get()).sum();
        let mut population = Population::with_capacity(target_size);
        let mut remaining = target_size;

        for (i, (operator, weight)) in self.operators.iter().enumerate() {
            let target = if i == self.operators.len() - 1 {
                remaining
            } else {
                let t = (weight.get() as usize * target_size) / total_weight as usize;
                remaining -= t;
                t
            };

            let fill = Fill::from_fixed_size(operator, target);
            let offspring = fill.apply(state, ctx);
            population.add_offspring(offspring);
        }

        Offspring::Multiple(population)
    }
}

macro_rules! impl_genetic_proportional {
    () => {};
    ($($Op:ident, $W:ident),+) => {
        impl<G, F, Fe, R, C, S, $($Op),*> GeneticOperator<G, F, Fe, R, C>
            for Proportional<( $(($Op, NonZero<u16>),)* ), S>
        where
            $($Op: GeneticOperator<G, F, Fe, R, C>),*,
            S: GetSize<G, F>,
        {
            #[allow(unused_assignments, non_snake_case)]
            fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
                let target_size = self.size.get_size(state);
                let ( $( ($Op, $W), )* ) = &self.operators;
                let total_weight: u16 = 0 $( + $W.get() )*;
                let count: usize = 0 $( + { let _ = $W; 1 } )*;
                let mut population = Population::with_capacity(target_size);
                let mut remaining = target_size;
                let mut current: usize = 0;

                $(
                    current += 1;
                    let target = if current == count {
                        remaining
                    } else {
                        let t = ($W.get() as usize * target_size) / total_weight as usize;
                        remaining -= t;
                        t
                    };

                    let fill = Fill::from_fixed_size($Op, target);
                    let offspring = fill.apply(state, ctx);
                    population.add_offspring(offspring);
                )*

                Offspring::Multiple(population)
            }
        }

        impl_genetic_proportional_recurse!($($Op, $W),+);
    };
}

macro_rules! impl_genetic_proportional_recurse {
    ($OpHead:ident, $WHead:ident, $($OpRest:ident, $WRest:ident),+) => {
        impl_genetic_proportional!($($OpRest, $WRest),+);
    };
    ($OpHead:ident, $WHead:ident) => {};
}

impl_genetic_proportional!(
    O1, W1, O2, W2, O3, W3, O4, W4, O5, W5, O6, W6, O7, W7, O8, W8, O9, W9, O10, W10, O11, W11,
    O12, W12, O13, W13, O14, W14, O15, W15, O16, W16
);
