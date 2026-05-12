//! The [`GrammarDef`] trait abstracting grammar access for the mapper.

use crate::ge::grammar::{Grammar, Symbol};

/// Trait abstracting grammar access for the mapper.
///
/// Both the built-in [`Grammar`] struct and user-defined types can implement
/// this to work with [`map()`](super::mapper::map).
pub trait GrammarDef {
    /// The type used to identify symbols.
    type Symbol: Copy;

    /// The terminal value type.
    type Terminal: Clone;

    /// Returns the start symbol.
    fn start(&self) -> Self::Symbol;

    /// Returns the number of productions for a non-terminal symbol.
    /// Returns 0 for terminals.
    fn num_productions(&self, symbol: Self::Symbol) -> usize;

    /// Returns the symbols in a specific production of a rule.
    fn production(&self, symbol: Self::Symbol, index: usize) -> &[Self::Symbol];

    /// Returns true if the symbol is a terminal.
    fn is_terminal(&self, symbol: Self::Symbol) -> bool {
        self.num_productions(symbol) == 0
    }

    /// Returns the terminal value for a terminal symbol.
    fn terminal_value(&self, symbol: Self::Symbol) -> Self::Terminal;
}

impl<T: Clone> GrammarDef for Grammar<T> {
    type Symbol = Symbol;
    type Terminal = T;

    fn start(&self) -> Symbol {
        Symbol::NonTerminal(self.start())
    }

    fn num_productions(&self, symbol: Symbol) -> usize {
        match symbol {
            Symbol::Terminal(_) => 0,
            Symbol::NonTerminal(i) => self.rules()[i].productions.len(),
        }
    }

    fn production(&self, symbol: Symbol, index: usize) -> &[Symbol] {
        match symbol {
            Symbol::NonTerminal(i) => &self.rules()[i].productions[index],
            Symbol::Terminal(_) => &[],
        }
    }

    fn is_terminal(&self, symbol: Symbol) -> bool {
        matches!(symbol, Symbol::Terminal(_))
    }

    fn terminal_value(&self, symbol: Symbol) -> T {
        match symbol {
            Symbol::Terminal(idx) => self.terminals()[idx].clone(),
            _ => panic!("terminal_value called on non-terminal"),
        }
    }
}
