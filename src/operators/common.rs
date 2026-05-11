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
    let point1 = rng.random_range(0..p1.len());
    let point2 = rng.random_range(0..p2.len());

    let child1 = [&p1[..point1], &p2[point2..]].concat();
    let child2 = [&p2[..point2], &p1[point1..]].concat();

    (child1, child2)
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
