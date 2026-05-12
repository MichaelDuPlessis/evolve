use evolve::fitness::{FitnessEvaluator, GeFitness};
use evolve::grammar;
use evolve::grammar::grammar_def::GrammarDef;
use evolve::phenotype::Phenotype;
use evolve::phenotype::bytecode::{Bytecode, BytecodeBuilder, Instruction};

grammar! {
    grammar Grammar;
    symbol Symbol;
    start Expr;
    Expr => [Expr, Expr, BinOp] | [Val];
    BinOp => [Add] | [Sub];
    Val => [X] | [One];
}

impl Instruction for Symbol {
    type Value = f64;
    type Input = f64;

    fn execute(&self, stack: &mut Vec<f64>, input: &f64) {
        match self {
            Symbol::X => stack.push(*input),
            Symbol::One => stack.push(1.0),
            Symbol::Add => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a + b);
            }
            Symbol::Sub => {
                let b = stack.pop().unwrap_or(0.0);
                let a = stack.pop().unwrap_or(0.0);
                stack.push(a - b);
            }
            _ => {}
        }
    }
}

#[test]
fn macro_grammar_def_works() {
    let g = Grammar;
    assert!(!g.is_terminal(g.start()));
    assert_eq!(g.num_productions(g.start()), 2);
}

#[test]
fn macro_grammar_maps_to_bytecode() {
    let fitness = GeFitness::<Grammar, u8, f64, _, BytecodeBuilder<Symbol>>::new(
        Grammar,
        0,
        |p: &Bytecode<Symbol>| p.run(&5.0),
        f64::NAN,
    );
    // Codons: 1 % 2 = 1 → Val, then 0 % 2 = 0 → X
    let result = fitness.evaluate(&vec![1u8, 0]);
    assert_eq!(result, 5.0);
}

#[test]
fn macro_grammar_runs_expression() {
    let fitness = GeFitness::<Grammar, u8, f64, _, BytecodeBuilder<Symbol>>::new(
        Grammar,
        0,
        |p: &Bytecode<Symbol>| p.run(&3.0),
        f64::NAN,
    );
    // Codons: 0 → Expr Expr BinOp (binary)
    //         1 → Val (left)
    //         0 → X (left val)
    //         1 → Val (right)
    //         1 → One (right val)
    //         0 → Add (binop)
    // Result: push x, push 1, add → x + 1
    let result = fitness.evaluate(&vec![0u8, 1, 0, 1, 1, 0]);
    assert_eq!(result, 4.0); // 3 + 1 = 4
}

#[test]
fn macro_grammar_with_ge_fitness() {
    let fitness = GeFitness::<Grammar, u8, f64, _, BytecodeBuilder<Symbol>>::new(
        Grammar,
        3,
        |program: &Bytecode<Symbol>| {
            // Test on x = 2: target is x + 1 = 3
            let result = program.run(&2.0);
            (result - 3.0).abs()
        },
        f64::MAX,
    );

    // Codons that produce x + 1
    let codons = vec![0u8, 1, 0, 1, 1, 0];
    assert_eq!(fitness.evaluate(&codons), 0.0);
}
