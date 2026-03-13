/// Trait to evaluate fitness
pub trait FitnessEvaluator<G, F> {
    fn evaluate(&self, genome: &G) -> F;
}
