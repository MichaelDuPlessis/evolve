/// Trait to evaluate fitness
pub trait FitnessEvaluator<G, F> {
    fn evaluate(&self, genome: &G) -> F;
}

impl<G, F, E> FitnessEvaluator<G, F> for E
where
    E: Fn(&G) -> F,
{
    fn evaluate(&self, genome: &G) -> F {
        self(genome)
    }
}

pub trait FitnessComparator<F> {
    fn is_better(&self, f1: &F, f2: &F) -> bool;
}

impl<F, C> FitnessComparator<F> for C
where
    C: Fn(&F, &F) -> bool,
{
    fn is_better(&self, f1: &F, f2: &F) -> bool {
        self(f1, f2)
    }
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
