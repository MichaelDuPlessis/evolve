//! Grammar representation and builder for grammatical evolution.

pub mod grammar_def;
pub mod mapper;

use std::collections::HashMap;
use std::hash::Hash;

/// A symbol in a grammar production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symbol {
    /// A terminal symbol. The usize is an index into `Grammar::terminals`.
    Terminal(usize),
    /// A non-terminal symbol. The usize is an index into `Grammar::rules`.
    NonTerminal(usize),
}

/// A rule in the grammar (a non-terminal and its productions).
#[derive(Debug, Clone)]
pub struct Rule {
    pub(crate) productions: Vec<Vec<Symbol>>,
}

/// A context-free grammar for grammatical evolution.
///
/// # Examples
///
/// ```
/// use evolve::grammar::Grammar;
///
/// let grammar = Grammar::builder()
///     .rule("expr", &[&["expr", "op", "expr"], &["x"]])
///     .rule("op", &[&["+"], &["-"]])
///     .start("expr")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct Grammar<T> {
    rules: Vec<Rule>,
    terminals: Vec<T>,
    start: usize,
}

impl<T: Eq + Hash + Clone> Grammar<T> {
    /// Creates a new [`GrammarBuilder`].
    pub fn builder() -> GrammarBuilder<T> {
        GrammarBuilder::new()
    }
}

impl<T> Grammar<T> {
    /// Returns the terminal value for the given index.
    pub fn terminal_value(&self, idx: usize) -> &T {
        &self.terminals[idx]
    }

    /// Returns the rules.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Returns the terminals.
    pub fn terminals(&self) -> &[T] {
        &self.terminals
    }

    /// Returns the start rule index.
    pub fn start(&self) -> usize {
        self.start
    }
}

/// Builder for ergonomic grammar construction.
///
/// # Examples
///
/// ```
/// use evolve::grammar::GrammarBuilder;
///
/// let grammar = GrammarBuilder::new()
///     .rule("expr", &[&["expr", "+", "expr"], &["x"]])
///     .start("expr")
///     .build();
/// ```
pub struct GrammarBuilder<T> {
    rules: Vec<(T, Vec<Vec<T>>)>,
    start: Option<T>,
}

impl<T: Eq + Hash + Clone> GrammarBuilder<T> {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            start: None,
        }
    }

    /// Adds a rule. Symbols matching any rule name are classified as non-terminals
    /// at [`build()`](Self::build) time; everything else is a terminal.
    pub fn rule(mut self, name: T, productions: &[&[T]]) -> Self {
        let prods = productions.iter().map(|p| p.to_vec()).collect();
        self.rules.push((name, prods));
        self
    }

    /// Sets the start symbol.
    pub fn start(mut self, name: T) -> Self {
        self.start = Some(name);
        self
    }

    /// Builds the grammar.
    ///
    /// # Panics
    ///
    /// Panics if the start symbol is unset or doesn't match a rule name.
    pub fn build(self) -> Grammar<T> {
        let start_name = self.start.expect("start symbol must be set");

        // Assign rule indices (order of definition).
        let mut rule_index: HashMap<&T, usize> = HashMap::new();
        for (i, (name, _)) in self.rules.iter().enumerate() {
            rule_index.insert(name, i);
        }

        assert!(
            rule_index.contains_key(&start_name),
            "start symbol must match a rule name"
        );

        // Collect terminals.
        let mut terminal_index: HashMap<&T, usize> = HashMap::new();
        let mut terminals: Vec<T> = Vec::new();
        for (_, prods) in &self.rules {
            for prod in prods {
                for sym in prod {
                    if !rule_index.contains_key(sym) && !terminal_index.contains_key(sym) {
                        terminal_index.insert(sym, terminals.len());
                        terminals.push(sym.clone());
                    }
                }
            }
        }

        // Build rules with Symbol references.
        let rules: Vec<Rule> = self
            .rules
            .iter()
            .map(|(_name, prods)| {
                let productions = prods
                    .iter()
                    .map(|p| {
                        p.iter()
                            .map(|s| {
                                if let Some(&idx) = rule_index.get(s) {
                                    Symbol::NonTerminal(idx)
                                } else {
                                    Symbol::Terminal(terminal_index[s])
                                }
                            })
                            .collect()
                    })
                    .collect();
                Rule { productions }
            })
            .collect();

        let start = rule_index[&start_name];

        Grammar {
            rules,
            terminals,
            start,
        }
    }
}

impl<T: Eq + Hash + Clone> Default for GrammarBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn builder_constructs_grammar() {
        let grammar = Grammar::builder()
            .rule("expr", &[&["expr", "op", "expr"], &["x"]])
            .rule("op", &[&["+"], &["-"]])
            .start("expr")
            .build();

        assert_eq!(grammar.start, 0);
        assert_eq!(grammar.rules[grammar.start].productions.len(), 2);
    }

    #[test]
    fn symbol_classification() {
        let grammar = Grammar::builder()
            .rule("expr", &[&["expr", "+", "x"]])
            .start("expr")
            .build();

        // "expr" is a rule, "+" and "x" are terminals
        assert_eq!(grammar.rules.len(), 1);
        assert_eq!(grammar.terminals.len(), 2);
        assert!(grammar.terminals.contains(&"+"));
        assert!(grammar.terminals.contains(&"x"));
    }

    #[test]
    fn terminal_value_lookup() {
        let grammar = Grammar::builder()
            .rule("s", &[&["hello"]])
            .start("s")
            .build();

        assert_eq!(grammar.terminal_value(0), &"hello");
    }

    #[test]
    #[should_panic(expected = "start symbol must be set")]
    fn missing_start_panics() {
        Grammar::<&str>::builder().rule("s", &[&["x"]]).build();
    }

    #[test]
    #[should_panic(expected = "start symbol must match a rule name")]
    fn invalid_start_panics() {
        Grammar::builder()
            .rule("s", &[&["x"]])
            .start("missing")
            .build();
    }
}
