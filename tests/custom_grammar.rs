use evolve::fitness::{FitnessEvaluator, GeFitness};
use evolve::grammar::grammar_def::GrammarDef;
use evolve::phenotype::{Event, PhenotypeBuilder};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Sym {
    Expr,
    Op,
    Val,
    Add,
    Sub,
    X,
    One,
}

// Productions stored as static slices
static EXPR_0: [Sym; 3] = [Sym::Expr, Sym::Expr, Sym::Op]; // [Expr, Expr, Op]
static EXPR_1: [Sym; 1] = [Sym::Val];
static OP_0: [Sym; 1] = [Sym::Add];
static OP_1: [Sym; 1] = [Sym::Sub];
static VAL_0: [Sym; 1] = [Sym::X];
static VAL_1: [Sym; 1] = [Sym::One];

struct MathGrammar;

impl GrammarDef for MathGrammar {
    type Symbol = Sym;
    type Terminal = Sym;

    fn start(&self) -> Sym {
        Sym::Expr
    }

    fn num_productions(&self, symbol: Sym) -> usize {
        match symbol {
            Sym::Expr => 2,
            Sym::Op => 2,
            Sym::Val => 2,
            _ => 0, // terminals
        }
    }

    fn production(&self, symbol: Sym, index: usize) -> &[Sym] {
        match (symbol, index) {
            (Sym::Expr, 0) => &EXPR_0,
            (Sym::Expr, 1) => &EXPR_1,
            (Sym::Op, 0) => &OP_0,
            (Sym::Op, 1) => &OP_1,
            (Sym::Val, 0) => &VAL_0,
            (Sym::Val, 1) => &VAL_1,
            _ => &[],
        }
    }

    fn is_terminal(&self, symbol: Sym) -> bool {
        matches!(symbol, Sym::Add | Sym::Sub | Sym::X | Sym::One)
    }

    fn terminal_value(&self, symbol: Sym) -> Sym {
        symbol
    }
}

// Stack machine phenotype
struct StackMachine(Vec<Sym>);

impl StackMachine {
    fn run(&self, x: &f64) -> f64 {
        let mut stack: Vec<f64> = Vec::new();
        for &sym in &self.0 {
            match sym {
                Sym::X => stack.push(*x),
                Sym::One => stack.push(1.0),
                Sym::Add => {
                    let b = stack.pop().unwrap_or(0.0);
                    let a = stack.pop().unwrap_or(0.0);
                    stack.push(a + b);
                }
                Sym::Sub => {
                    let b = stack.pop().unwrap_or(0.0);
                    let a = stack.pop().unwrap_or(0.0);
                    stack.push(a - b);
                }
                _ => {}
            }
        }
        stack.pop().unwrap_or(0.0)
    }
}

// Builder collects terminals in postfix order
#[derive(Default)]
struct StackBuilder(Vec<Sym>);

impl PhenotypeBuilder<Sym> for StackBuilder {
    type Output = StackMachine;

    fn push(&mut self, event: Event<Sym>) {
        if let Event::Terminal(t) = event {
            self.0.push(t);
        }
    }

    fn finish(self) -> StackMachine {
        StackMachine(self.0)
    }
}

// --- Tests ---

#[test]
fn grammar_def_start() {
    assert_eq!(MathGrammar.start(), Sym::Expr);
}

#[test]
fn grammar_def_num_productions() {
    assert_eq!(MathGrammar.num_productions(Sym::Expr), 2);
    assert_eq!(MathGrammar.num_productions(Sym::Op), 2);
    assert_eq!(MathGrammar.num_productions(Sym::Val), 2);
    assert_eq!(MathGrammar.num_productions(Sym::Add), 0);
    assert_eq!(MathGrammar.num_productions(Sym::X), 0);
}

#[test]
fn grammar_def_production() {
    assert_eq!(
        MathGrammar.production(Sym::Expr, 0),
        &[Sym::Expr, Sym::Expr, Sym::Op]
    );
    assert_eq!(MathGrammar.production(Sym::Expr, 1), &[Sym::Val]);
    assert_eq!(MathGrammar.production(Sym::Op, 0), &[Sym::Add]);
    assert_eq!(MathGrammar.production(Sym::Op, 1), &[Sym::Sub]);
    assert_eq!(MathGrammar.production(Sym::Val, 0), &[Sym::X]);
    assert_eq!(MathGrammar.production(Sym::Val, 1), &[Sym::One]);
}

#[test]
fn grammar_def_is_terminal() {
    assert!(!MathGrammar.is_terminal(Sym::Expr));
    assert!(!MathGrammar.is_terminal(Sym::Op));
    assert!(!MathGrammar.is_terminal(Sym::Val));
    assert!(MathGrammar.is_terminal(Sym::Add));
    assert!(MathGrammar.is_terminal(Sym::Sub));
    assert!(MathGrammar.is_terminal(Sym::X));
    assert!(MathGrammar.is_terminal(Sym::One));
}

#[test]
fn grammar_def_terminal_value() {
    assert_eq!(MathGrammar.terminal_value(Sym::Add), Sym::Add);
    assert_eq!(MathGrammar.terminal_value(Sym::X), Sym::X);
}

#[test]
fn phenotype_run_single_val() {
    // Just "X" → should return x
    let machine = StackMachine(vec![Sym::X]);
    assert_eq!(machine.run(&3.0), 3.0);

    // Just "One" → should return 1.0
    let machine = StackMachine(vec![Sym::One]);
    assert_eq!(machine.run(&5.0), 1.0);
}

#[test]
fn phenotype_run_addition() {
    // X X Add → x + x
    let machine = StackMachine(vec![Sym::X, Sym::X, Sym::Add]);
    assert_eq!(machine.run(&3.0), 6.0);
}

#[test]
fn phenotype_run_subtraction() {
    // X One Sub → x - 1
    let machine = StackMachine(vec![Sym::X, Sym::One, Sym::Sub]);
    assert_eq!(machine.run(&5.0), 4.0);
}

#[test]
fn ge_fitness_produces_valid_phenotype() {
    // Codons: [1, 0] → Expr picks prod 1 (Val), Val picks prod 0 (X)
    // Result: StackMachine([X]), run(2.0) = 2.0
    let ge = GeFitness::<_, u8, f64, _, StackBuilder, _>::new(
        MathGrammar,
        3,
        |p: &StackMachine| p.run(&2.0),
        -999.0,
    );
    assert_eq!(ge.evaluate(&vec![1u8, 0]), 2.0);
}

#[test]
fn ge_fitness_complex_expression() {
    // Codons: [0, 1, 0, 0, 1, 0]
    // Expr → prod 0: [Expr, Expr, Op]
    //   Expr → codon 1 % 2 = 1 → prod 1: [Val]
    //     Val → codon 0 % 2 = 0 → [X]
    //   Expr → codon 0 % 2 = 0 → prod 0: [Expr, Expr, Op] ... but let's use simpler codons
    //
    // Let's use: [0, 1, 1, 0, 1] to get Expr(Expr,Expr,Op) → Val(X), Val(One), Add
    // Expr → prod 0 (codon 0%2=0): [Expr, Expr, Op]
    //   Expr → prod 1 (codon 1%2=1): [Val]
    //     Val → prod 1 (codon 1%2=1): [One]
    //   Expr → prod 0 (codon 0%2=0): [Val] ... wait, 0%2=0 → [Expr,Expr,Op]
    //
    // Simpler: codons [0, 1, 0, 1, 0]
    // Expr → codon 0%2=0: [Expr, Expr, Op]
    //   Expr → codon 1%2=1: [Val]
    //     Val → codon 0%2=0: [X]
    //   Expr → codon 1%2=1: [Val]
    //     Val → codon 0%2=0: [X]
    //   Op → codon (wraps)...
    // Actually let me just use enough codons:
    // [0, 1, 0, 1, 1, 0]
    // Expr → 0%2=0: [Expr, Expr, Op]
    //   Expr → 1%2=1: [Val]
    //     Val → 0%2=0: [X]
    //   Expr → 1%2=1: [Val]
    //     Val → 1%2=1: [One]
    //   Op → 0%2=0: [Add]
    // Terminals in order: X, One, Add → StackMachine([X, One, Add]) → x + 1
    let ge = GeFitness::<_, u8, f64, _, StackBuilder, _>::new(
        MathGrammar,
        3,
        |p: &StackMachine| p.run(&5.0),
        -999.0,
    );
    assert_eq!(ge.evaluate(&vec![0u8, 1, 0, 1, 1, 0]), 6.0); // x + 1 = 5 + 1 = 6
}

#[test]
fn ge_fitness_invalid_returns_penalty() {
    // All-recursive codons with 0 wraps should exhaust
    let ge = GeFitness::<_, u8, f64, _, StackBuilder, _>::new(
        MathGrammar,
        0,
        |p: &StackMachine| p.run(&1.0),
        -999.0,
    );
    // Single codon 0 picks recursive [Expr, Expr, Op] repeatedly
    assert_eq!(ge.evaluate(&vec![0u8]), -999.0);
}
