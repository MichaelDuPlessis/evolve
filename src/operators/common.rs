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

/// Performs uniform crossover on two fixed-length genomes.
pub(crate) fn uniform_crossover<T: Clone, const N: usize>(
    p1: &[T; N],
    p2: &[T; N],
    rng: &mut impl Rng,
) -> ([T; N], [T; N]) {
    let mut child1 = p1.clone();
    let mut child2 = p2.clone();

    for i in 0..N {
        if rng.random::<bool>() {
            child1[i].clone_from(&p2[i]);
            child2[i].clone_from(&p1[i]);
        }
    }

    (child1, child2)
}

/// Performs uniform crossover on two variable-length genomes.
pub(crate) fn uniform_crossover_vec<T: Clone>(
    p1: &[T],
    p2: &[T],
    rng: &mut impl Rng,
) -> (Vec<T>, Vec<T>) {
    let min_len = p1.len().min(p2.len());
    let mut child1 = p1.to_vec();
    let mut child2 = p2.to_vec();

    for i in 0..min_len {
        if rng.random::<bool>() {
            child1[i].clone_from(&p2[i]);
            child2[i].clone_from(&p1[i]);
        }
    }

    (child1, child2)
}

/// Performs two-point crossover on two fixed-length genomes.
pub(crate) fn two_point_crossover<T: Clone, const N: usize>(
    p1: &[T; N],
    p2: &[T; N],
    rng: &mut impl Rng,
) -> ([T; N], [T; N]) {
    let mut a = rng.random_range(0..N);
    let mut b = rng.random_range(0..N);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let mut child1 = p1.clone();
    let mut child2 = p2.clone();

    child1[a..b].clone_from_slice(&p2[a..b]);
    child2[a..b].clone_from_slice(&p1[a..b]);

    (child1, child2)
}

/// Performs two-point crossover on two variable-length genomes.
pub(crate) fn two_point_crossover_vec<T: Clone>(
    p1: &[T],
    p2: &[T],
    rng: &mut impl Rng,
) -> (Vec<T>, Vec<T>) {
    if p1.len() < 2 || p2.len() < 2 {
        return (p1.to_vec(), p2.to_vec());
    }

    let mut a1 = rng.random_range(0..p1.len());
    let mut b1 = rng.random_range(0..p1.len());
    if a1 > b1 { std::mem::swap(&mut a1, &mut b1); }

    let mut a2 = rng.random_range(0..p2.len());
    let mut b2 = rng.random_range(0..p2.len());
    if a2 > b2 { std::mem::swap(&mut a2, &mut b2); }

    // child1 = p1[..a1] + p2[a2..b2] + p1[b1..]
    let mut child1 = Vec::with_capacity(a1 + (b2 - a2) + (p1.len() - b1));
    child1.extend_from_slice(&p1[..a1]);
    child1.extend_from_slice(&p2[a2..b2]);
    child1.extend_from_slice(&p1[b1..]);

    // child2 = p2[..a2] + p1[a1..b1] + p2[b2..]
    let mut child2 = Vec::with_capacity(a2 + (b1 - a1) + (p2.len() - b2));
    child2.extend_from_slice(&p2[..a2]);
    child2.extend_from_slice(&p1[a1..b1]);
    child2.extend_from_slice(&p2[b2..]);

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
