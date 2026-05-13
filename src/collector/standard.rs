//! Standard collector that records best fitness and generation durations.

use crate::collector::Collector;
use crate::core::population::Population;
use crate::core::state::State;
use crate::fitness::{FitnessComparator, FitnessEvaluator};
use std::time::{Duration, Instant};

/// Collects best fitness per generation and timing data during a run.
pub struct Standard<F> {
    start_time: Option<Instant>,
    best_fitness: Vec<F>,
    generation_durations: Vec<Duration>,
    last_tick: Instant,
}

impl<F> Default for Standard<F> {
    fn default() -> Self {
        Self {
            start_time: None,
            best_fitness: Vec::new(),
            generation_durations: Vec::new(),
            last_tick: Instant::now(),
        }
    }
}

/// The final result of an algorithm run collected by [`Standard`].
#[derive(Debug)]
pub struct RunResult<G, F> {
    population: Population<G, F>,
    generations: usize,
    total_duration: Duration,
    best_fitness: Vec<F>,
    generation_durations: Vec<Duration>,
}

impl<G, F> RunResult<G, F> {
    /// Returns a reference to the final population.
    pub fn population(&self) -> &Population<G, F> {
        &self.population
    }

    /// Returns the number of generations completed.
    pub fn generations(&self) -> usize {
        self.generations
    }

    /// Returns the total duration of the run.
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Returns the best fitness recorded at each generation.
    pub fn best_fitness(&self) -> &[F] {
        &self.best_fitness
    }

    /// Returns the duration of each generation.
    pub fn generation_durations(&self) -> &[Duration] {
        &self.generation_durations
    }

    /// Consumes the result and returns the final population.
    pub fn into_population(self) -> Population<G, F> {
        self.population
    }

    /// Returns the data for a specific generation (0-indexed).
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn generation(&self, index: usize) -> Option<GenerationRecord<'_, F>> {
        if index >= self.best_fitness.len() {
            return None;
        }
        Some(GenerationRecord {
            best_fitness: &self.best_fitness[index],
            duration: self.generation_durations[index],
        })
    }
}

/// A view into a single generation's recorded data.
#[derive(Debug, Clone, Copy)]
pub struct GenerationRecord<'a, F> {
    best_fitness: &'a F,
    duration: Duration,
}

impl<'a, F> GenerationRecord<'a, F> {
    /// The best fitness value in this generation.
    pub fn best_fitness(&self) -> &F {
        self.best_fitness
    }

    /// The wall-clock duration of this generation.
    pub fn duration(&self) -> Duration {
        self.duration
    }
}

impl<G, F, Fe, C> Collector<G, F, Fe, C> for Standard<F>
where
    F: Clone + PartialOrd,
    Fe: FitnessEvaluator<G, F>,
    C: FitnessComparator<F>,
{
    type Result = RunResult<G, F>;

    fn on_start(&mut self, _state: &State<G, F>, _fe: &Fe, _cmp: &C) {
        let now = Instant::now();
        self.start_time = Some(now);
        self.last_tick = now;
    }

    fn on_generation(&mut self, state: &State<G, F>, fe: &Fe, cmp: &C) {
        let now = Instant::now();
        self.generation_durations.push(now - self.last_tick);
        self.last_tick = now;

        let best = state.population().best(fe, cmp);
        self.best_fitness.push(best.fitness(fe).clone());
    }

    fn finalize(self, state: State<G, F>) -> Self::Result {
        let total_duration = self
            .start_time
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);

        RunResult {
            generations: state.generation(),
            population: state.into_population(),
            total_duration,
            best_fitness: self.best_fitness,
            generation_durations: self.generation_durations,
        }
    }
}
