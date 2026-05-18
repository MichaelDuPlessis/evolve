use crate::{
    core::{context::Context, offspring::Offspring, state::State},
    operators::GeneticOperator,
};

/// Applies one of two operators based on a predicate.
///
/// The predicate receives the current state and returns `true` for operator A
/// or `false` for operator B. Useful for adaptive strategies (e.g., switch
/// operators based on generation number or fitness convergence).
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::combinator::Conditional;
/// use evolve::operators::sequential::mutation::RandomReset;
/// use evolve::operators::sequential::identity::Identity;
/// use evolve::core::state::State;
///
/// // Apply mutation only after generation 10
/// let op = Conditional::new(
///     RandomReset::<u8>::new(),
///     Identity,
///     |state: &State<[u8; 4], i32>| state.generation() >= 10,
/// );
/// ```
#[derive(Debug, Clone)]
pub struct Conditional<A, B, P> {
    a: A,
    b: B,
    predicate: P,
}

impl<A, B, P> Conditional<A, B, P> {
    /// Creates a new `Conditional`.
    ///
    /// Applies `a` when the predicate returns `true`, `b` otherwise.
    pub fn new(a: A, b: B, predicate: P) -> Self {
        Self { a, b, predicate }
    }
}

impl<G, F, Fe, R, C, A, B, P> GeneticOperator<G, F, Fe, R, C> for Conditional<A, B, P>
where
    A: GeneticOperator<G, F, Fe, R, C>,
    B: GeneticOperator<G, F, Fe, R, C>,
    P: Fn(&State<G, F>) -> bool,
{
    fn apply(&self, state: &State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        if (self.predicate)(state) {
            self.a.apply(state, ctx)
        } else {
            self.b.apply(state, ctx)
        }
    }

    fn transform(&self, state: State<G, F>, ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        if (self.predicate)(&state) {
            self.a.transform(state, ctx)
        } else {
            self.b.transform(state, ctx)
        }
    }
}
