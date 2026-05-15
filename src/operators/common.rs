//! Shared operator logic used by both sequential and parallel implementations.

use crate::random::Randomizable;
use rand::{Rng, RngExt};

/// Performs single-point crossover on two genomes, returning two children.
pub(crate) fn single_point_crossover<T: Clone, const N: usize>(
    p1: &[T; N],
    p2: &[T; N],
    rng: &mut impl Rng,
) -> ([T; N], [T; N]) {
    let point = rng.random_range(0..N);

    let mut child1 = p1.clone();
    let mut child2 = p2.clone();

    child1[point..N].clone_from_slice(&p2[point..N]);
    child2[point..N].clone_from_slice(&p1[point..N]);

    (child1, child2)
}

/// Performs single-point crossover on two variable-length genomes, returning two children.
pub(crate) fn single_point_crossover_vec<T: Clone>(
    p1: &[T],
    p2: &[T],
    rng: &mut impl Rng,
) -> (Vec<T>, Vec<T>) {
    if p1.len() < 2 || p2.len() < 2 {
        let mut c1: vecpool::PoolVec<T> = vecpool::with_capacity(p1.len());
        c1.extend_from_slice(p1);
        let mut c2: vecpool::PoolVec<T> = vecpool::with_capacity(p2.len());
        c2.extend_from_slice(p2);
        return (c1.into_vec(), c2.into_vec());
    }

    let ratio: f64 = rng.random();
    let point1 = ((ratio * p1.len() as f64) as usize).clamp(1, p1.len() - 1);
    let point2 = ((ratio * p2.len() as f64) as usize).clamp(1, p2.len() - 1);

    let mut child1: vecpool::PoolVec<T> = vecpool::with_capacity(point1 + p2.len() - point2);
    child1.extend_from_slice(&p1[..point1]);
    child1.extend_from_slice(&p2[point2..]);

    let mut child2: vecpool::PoolVec<T> = vecpool::with_capacity(point2 + p1.len() - point1);
    child2.extend_from_slice(&p2[..point2]);
    child2.extend_from_slice(&p1[point1..]);

    (child1.into_vec(), child2.into_vec())
}

/// Mutates a genome by replacing a random gene with a new random value.
pub(crate) fn random_reset_mutate<G, T, R>(genome: &G, rng: &mut R) -> G
where
    G: Clone + AsMut<[T]>,
    T: Randomizable<R>,
    R: Rng,
{
    let mut new_genome = genome.clone();
    let genes = new_genome.as_mut();
    let gene_index = rng.random_range(0..genes.len());
    genes[gene_index] = T::random(rng);
    new_genome
}
