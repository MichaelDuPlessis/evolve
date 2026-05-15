//! Bytecode phenotype — a flat list of instructions executed on a stack machine.

use crate::phenotype::{Event, PhenotypeBuilder};

/// Trait for types that can be executed as stack machine instructions.
///
/// # Examples
///
/// ```
/// use evolve::phenotype::bytecode::{Instruction, Bytecode, BytecodeBuilder};
/// use evolve::phenotype::{PhenotypeBuilder, Event};
///
/// #[derive(Clone, PartialEq, Eq, Hash)]
/// enum Op { Push(i32), Add }
///
/// impl Instruction for Op {
///     type Value = i32;
///     type Input = ();
///     fn execute(&self, stack: &mut Vec<i32>, _: &()) {
///         match self {
///             Op::Push(n) => stack.push(*n),
///             Op::Add => {
///                 let b = stack.pop().unwrap_or(0);
///                 let a = stack.pop().unwrap_or(0);
///                 stack.push(a + b);
///             }
///         }
///     }
/// }
///
/// let program = Bytecode::new(vec![Op::Push(2), Op::Push(3), Op::Add]);
/// assert_eq!(program.run(&()), 5);
/// ```
pub trait Instruction {
    /// The value type on the stack.
    type Value: Default;
    /// The input type passed to the program.
    type Input;

    /// Execute this instruction, modifying the stack.
    fn execute(&self, stack: &mut Vec<Self::Value>, input: &Self::Input);
}

/// A bytecode program — a flat list of instructions.
#[derive(Debug)]
pub struct Bytecode<T>(Vec<T>);

impl<T> Bytecode<T> {
    /// Creates a new bytecode program from a list of instructions.
    pub fn new(instructions: Vec<T>) -> Self {
        Self(instructions)
    }

    /// Returns the instructions.
    pub fn instructions(&self) -> &[T] {
        &self.0
    }
}

impl<T: Instruction> Bytecode<T> {
    /// Executes the bytecode program with the given input, returning the final stack value.
    pub fn run(&self, input: &T::Input) -> T::Value {
        let mut stack: vecpool::PoolVec<T::Value> = vecpool::PoolVec::new();
        for instruction in &self.0 {
            instruction.execute(&mut *stack, input);
        }
        stack.pop().unwrap_or_default()
    }
}

/// Builder that collects terminals into a [`Bytecode`] program.
#[derive(Debug)]
pub struct BytecodeBuilder<T>(vecpool::PoolVec<T>);

impl<T> Default for BytecodeBuilder<T> {
    fn default() -> Self {
        Self(vecpool::PoolVec::new())
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
        Bytecode(self.0.into_vec())
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
        let prog = Bytecode::new(vec![Op::InputX, Op::Push(3.0), Op::Add]);
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
        let prog: Bytecode<Op> = Bytecode::new(vec![]);
        assert_eq!(prog.run(&1.0), 0.0);
    }
}
