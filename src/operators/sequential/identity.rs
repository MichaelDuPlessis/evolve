use crate::core::{context::Context, offspring::Offspring, state::State};
use crate::operators::GeneticOperator;

/// No-op operator that passes the population through unchanged.
///
/// # Examples
///
/// ```
/// use evolve::operators::sequential::identity::Identity;
///
/// let op = Identity::new();
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Identity;

impl Identity {
    /// Creates a new `Identity` operator.
    pub fn new() -> Self {
        Self
    }
}

impl<G: Clone, F: Clone, Fe, R, C> GeneticOperator<G, F, Fe, R, C> for Identity {
    fn apply(&self, state: &State<G, F>, _ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        Offspring::Multiple(state.population().clone())
    }

    fn transform(&self, state: State<G, F>, _ctx: &mut Context<Fe, R, C>) -> Offspring<G, F> {
        Offspring::Multiple(state.into_population())
    }
}
