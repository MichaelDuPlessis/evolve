/// Trait to evaluate fitness
pub trait FitnessEvaluator<G, F> {
    fn evaluate(&self, genome: &G) -> F;
}

pub trait FitnessComparator<F>
where
    F: PartialOrd,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool;
}

pub struct Maximize;

impl<F> FitnessComparator<F> for Maximize
where
    F: PartialOrd,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        f1 > f2
    }
}

pub struct Minimize;

impl<F> FitnessComparator<F> for Minimize
where
    F: PartialOrd,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        f1 < f2
    }
}
