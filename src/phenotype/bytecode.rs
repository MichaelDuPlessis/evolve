//! Bytecode phenotype — a flat list of instructions executed on a stack machine.

use crate::phenotype::phenotype::{Event, Phenotype, PhenotypeBuilder};

/// Trait for types that can be executed as stack machine instructions.
pub trait Instruction {
    /// The value type on the stack.
    type Value: Default;
    /// The input type passed to the program.
    type Input;

    /// Execute this instruction, modifying the stack.
    fn execute(&self, stack: &mut Vec<Self::Value>, input: &Self::Input);
}

/// A bytecode program — a flat list of instructions.
pub struct Bytecode<T>(pub Vec<T>);

impl<T: Instruction> Phenotype for Bytecode<T> {
    type Input = T::Input;
    type Output = T::Value;

    fn run(&self, input: &Self::Input) -> Self::Output {
        let mut stack = Vec::new();
        for instruction in &self.0 {
            instruction.execute(&mut stack, input);
        }
        stack.pop().unwrap_or_default()
    }
}

/// Builder that collects terminals into a [`Bytecode`] program.
pub struct BytecodeBuilder<T>(Vec<T>);

impl<T> Default for BytecodeBuilder<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T: Instruction> PhenotypeBuilder<T> for BytecodeBuilder<T> {
    type Output = Bytecode<T>;

    fn push(&mut self, event: Event<T>) {
        if let Event::Terminal(t) = event {
            self.0.push(t);
        }
    }

    fn finish(self) -> Bytecode<T> {
        Bytecode(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    enum Op {
        Push(f64),
        Add,
        InputX,
    }

    impl Instruction for Op {
        type Value = f64;
        type Input = f64;

        fn execute(&self, stack: &mut Vec<f64>, input: &f64) {
            match self {
                Op::Push(v) => stack.push(*v),
                Op::Add => {
                    let b = stack.pop().unwrap_or_default();
                    let a = stack.pop().unwrap_or_default();
                    stack.push(a + b);
                }
                Op::InputX => stack.push(*input),
            }
        }
    }

    #[test]
    fn bytecode_runs_stack_machine() {
        let prog = Bytecode(vec![Op::InputX, Op::Push(3.0), Op::Add]);
        assert_eq!(prog.run(&2.0), 5.0);
    }

    #[test]
    fn builder_collects_terminals() {
        let mut builder = BytecodeBuilder::default();
        builder.push(Event::BeginRule);
        builder.push(Event::Terminal(Op::Push(1.0)));
        builder.push(Event::Terminal(Op::Push(2.0)));
        builder.push(Event::Terminal(Op::Add));
        builder.push(Event::EndRule);
        let prog = builder.finish();
        assert_eq!(prog.run(&0.0), 3.0);
    }

    #[test]
    fn empty_program_returns_default() {
        let prog: Bytecode<Op> = Bytecode(vec![]);
        assert_eq!(prog.run(&1.0), 0.0);
    }
}
