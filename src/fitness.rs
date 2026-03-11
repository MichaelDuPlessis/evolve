/// Trait to evaluate fitness
pub trait FitnessEvaluation<G, F> {
    fn evaluate(&self, genome: &G) -> F;
}
