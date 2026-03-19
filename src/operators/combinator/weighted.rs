use std::num::NonZero;

use rand::{Rng, RngExt};

use crate::{
    core::{context::Context, offspring::Offspring, state::State},
    operators::GeneticOperator,
};

/// Selects one operator per invocation based on assigned weights.
///
/// Each operator is paired with a `NonZero<u16>` weight. On each call, a random
/// roll determines which operator runs. Higher weights mean higher probability.
///
/// Accepts a tuple of up to 16 `(Operator, NonZero<u16>)` pairs.
///
/// # Examples
///
/// ```
/// use evolve::operators::combinator::Weighted;
/// use evolve::operators::mutation::RandomReset;
/// use std::num::NonZero;
///
/// // 75% chance of first operator, 25% chance of second
/// let op = Weighted::new((
///     (RandomReset::<u8>::new(), NonZero::new(3u16).unwrap()),
///     (RandomReset::<u8>::new(), NonZero::new(1u16).unwrap()),
/// ));
/// ```
#[derive(Debug)]
pub struct Weighted<O: ?Sized>(O);

impl<O> Weighted<O> {
    /// Creates a new `Weighted` from a tuple of `(operator, weight)` pairs.
    pub fn new(operators: O) -> Self {
        Self(operators)
    }
}

macro_rules! impl_genetic_weighted {
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
        impl_genetic_weighted_recurse!($($Op, $W),+);
    };
}

macro_rules! impl_genetic_weighted_recurse {
    ($OpHead:ident, $WHead:ident, $($OpRest:ident, $WRest:ident),+) => {
        impl_genetic_weighted !($($OpRest, $WRest),+);
    };
    ($OpHead:ident, $WHead:ident) => {};
}

impl_genetic_weighted!(
    O1, W1, O2, W2, O3, W3, O4, W4, O5, W5, O6, W6, O7, W7, O8, W8, O9, W9, O10, W10, O11, W11,
    O12, W12, O13, W13, O14, W14, O15, W15, O16, W16
);

impl<G, F, Fe, R, C, O> GeneticOperator<G, F, Fe, R, C> for Weighted<[(O, NonZero<u16>)]>
where
    O: GeneticOperator<G, F, Fe, R, C>,
    R: Rng,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        let total_weight: u16 = self.0.iter().map(|(_, w)| w.get()).sum();

        let mut roll = ctx.rng().random_range(0..total_weight);

        for (operator, weight) in &self.0 {
            let weight = weight.get();
            if roll < weight {
                return operator.apply(state, ctx);
            }
            roll -= weight;
        }

        unreachable!("Weighted selection failed")
    }
}
