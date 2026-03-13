use crate::core::individual::Individual;

/// The enum is used to determine whether the problem is a minimization or maxmization problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Goal {
    Maximize,
    Minimize,
}

impl Goal {
    /// Given a list of `Individuals` return the best one
    pub fn best<'a, G, F>(
        &self,
        individuals: impl IntoIterator<Item = &'a Individual<G, F>>,
    ) -> &'a Individual<G, F>
    where
        F: PartialOrd,
    {
        match self {
            Goal::Maximize => individuals
                .into_iter()
                .max_by(|a, b| a.fitness().partial_cmp(b.fitness()).unwrap())
                .expect("population cannot be empty"),
            Goal::Minimize => individuals
                .into_iter()
                .min_by(|a, b| a.fitness().partial_cmp(b.fitness()).unwrap())
                .expect("population cannot be empty"),
        }
    }

    /// Given two `Individuals` return true if the the first one is better than the second one.
    pub fn better<G, F>(&self, i1: Individual<G, F>, i2: Individual<G, F>) -> bool
    where
        F: PartialOrd,
    {
        let f1 = i1.fitness();
        let f2 = i2.fitness();

        match self {
            Goal::Maximize => f1 > f2,
            Goal::Minimize => f1 < f2,
        }
    }
}
