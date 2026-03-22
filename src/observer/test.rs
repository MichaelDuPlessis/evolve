use super::*;
use crate::core::{context::Context, individual::Individual, population::Population, state::State};
use crate::fitness::Maximize;

fn fe(g: &[u8; 1]) -> u8 {
    g[0]
}

fn make_state() -> State<[u8; 1], u8> {
    let pop = Population::from_iter(vec![Individual::new([1], &fe)]);
    State::new(pop, 0)
}

#[test]
fn custom_observer_receives_all_hooks() {
    struct Tracker {
        started: bool,
        generations: usize,
        ended: bool,
    }

    impl<G, F, Fe, R, C> Observer<G, F, Fe, R, C> for Tracker {
        fn on_start(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {
            self.started = true;
        }
        fn on_generation(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {
            self.generations += 1;
        }
        fn on_end(&mut self, _: &State<G, F>, _: &Context<Fe, R, C>) {
            self.ended = true;
        }
    }

    let mut rng = rand::rng();
    let ctx = Context::new(&(fe as fn(&[u8; 1]) -> u8), &mut rng, &Maximize);
    let state = make_state();

    let mut tracker = Tracker { started: false, generations: 0, ended: false };
    tracker.on_start(&state, &ctx);
    tracker.on_generation(&state, &ctx);
    tracker.on_generation(&state, &ctx);
    tracker.on_end(&state, &ctx);

    assert!(tracker.started);
    assert_eq!(tracker.generations, 2);
    assert!(tracker.ended);
}

#[test]
fn noop_observer_compiles() {
    let mut rng = rand::rng();
    let ctx = Context::new(&(fe as fn(&[u8; 1]) -> u8), &mut rng, &Maximize);
    let state = make_state();

    let mut noop = NoOp::new();
    noop.on_start(&state, &ctx);
    noop.on_generation(&state, &ctx);
    noop.on_end(&state, &ctx);
}

#[test]
fn default_methods_are_noop() {
    struct Empty;
    impl<G, F, Fe, R, C> Observer<G, F, Fe, R, C> for Empty {}

    let mut rng = rand::rng();
    let ctx = Context::new(&(fe as fn(&[u8; 1]) -> u8), &mut rng, &Maximize);
    let state = make_state();

    let mut empty = Empty;
    empty.on_start(&state, &ctx);
    empty.on_generation(&state, &ctx);
    empty.on_end(&state, &ctx);
}
