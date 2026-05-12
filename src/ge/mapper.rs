//! Codon-to-phenotype mapping.

use crate::ge::grammar_def::GrammarDef;
use crate::ge::phenotype::PhenotypeBuilder;

/// Trait bound for codon types. Converts a codon to a choice index.
pub trait Codon: Clone + Copy {
    /// Returns the codon value as a `usize`.
    fn as_usize(self) -> usize;
}

impl Codon for u8 {
    fn as_usize(self) -> usize {
        self as usize
    }
}
impl Codon for u16 {
    fn as_usize(self) -> usize {
        self as usize
    }
}
impl Codon for u32 {
    fn as_usize(self) -> usize {
        self as usize
    }
}
impl Codon for u64 {
    fn as_usize(self) -> usize {
        self as usize
    }
}
impl Codon for usize {
    fn as_usize(self) -> usize {
        self
    }
}

/// Maps a codon sequence through a grammar using a [`PhenotypeBuilder`].
///
/// Returns `None` if codons are exhausted after `max_wraps` wraps before all
/// non-terminals are expanded.
pub fn map<G: GrammarDef, C: Codon, B: PhenotypeBuilder<G>>(
    grammar: &G,
    codons: &[C],
    max_wraps: usize,
    mut builder: B,
) -> Option<B::Output> {
    let mut stack: Vec<G::Symbol> = vec![grammar.start()];
    // Track pending end_rule calls: each entry is remaining children count.
    let mut end_rule_stack: Vec<usize> = Vec::new();
    let mut codon_idx: usize = 0;

    while let Some(symbol) = stack.pop() {
        if grammar.is_terminal(symbol) {
            builder.terminal(symbol, grammar);
        } else {
            let n = grammar.num_productions(symbol);
            let choice = if n == 1 {
                0
            } else {
                if codons.is_empty() {
                    return None;
                }
                // Check if consuming this codon would exceed wraps.
                let current_wrap = codon_idx / codons.len();
                if current_wrap > max_wraps {
                    return None;
                }
                let c = codons[codon_idx % codons.len()].as_usize() % n;
                codon_idx += 1;
                c
            };

            builder.begin_rule(symbol, choice, grammar);
            let prod = grammar.production(symbol, choice);
            // Push in reverse for left-to-right expansion.
            for &s in prod.iter().rev() {
                stack.push(s);
            }
            end_rule_stack.push(prod.len());
            // Skip the decrement logic below since we just pushed children.
            continue;
        }

        // After processing a terminal, decrement parent counters.
        loop {
            match end_rule_stack.last_mut() {
                Some(count) => {
                    *count -= 1;
                    if *count == 0 {
                        end_rule_stack.pop();
                        builder.end_rule();
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
    }

    Some(builder.finish())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ge::grammar::{Grammar, Symbol};
    use crate::ge::phenotype::{Phenotype, PhenotypeBuilder};

    #[derive(Debug, PartialEq)]
    struct Program(String);
    impl Phenotype for Program {
        type Input = ();
        type Output = String;
        fn run(&self, _: &()) -> String { self.0.clone() }
    }

    #[derive(Default)]
    struct ProgramBuilder(String);
    impl PhenotypeBuilder<Grammar<&'static str>> for ProgramBuilder {
        type Output = Program;
        fn terminal(&mut self, symbol: Symbol, grammar: &Grammar<&'static str>) {
            if let Symbol::Terminal(idx) = symbol {
                self.0.push_str(grammar.terminal_value(idx).as_ref());
            }
        }
        fn begin_rule(&mut self, _: Symbol, _: usize, _: &Grammar<&'static str>) {}
        fn end_rule(&mut self) {}
        fn finish(self) -> Program { Program(self.0) }
    }

    fn expr_grammar() -> Grammar<&'static str> {
        Grammar::builder()
            .rule("expr", &[&["expr", "op", "expr"], &["x"], &["1"]])
            .rule("op", &[&["+"], &["-"]])
            .start("expr")
            .build()
    }

    #[test]
    fn simple_terminal() {
        let g = expr_grammar();
        let result = map(&g, &[1u8], 0, ProgramBuilder::default());
        assert_eq!(result.unwrap().run(&()), "x");
    }

    #[test]
    fn recursive_expansion() {
        let g = expr_grammar();
        let result = map(&g, &[0u8, 1, 0, 2], 0, ProgramBuilder::default());
        assert_eq!(result.unwrap().run(&()), "x+1");
    }

    #[test]
    fn wrapping() {
        let g = expr_grammar();
        let result = map(&g, &[0u8], 1, ProgramBuilder::default());
        assert_eq!(result, None);
    }

    #[test]
    fn single_production_no_codon() {
        let g = Grammar::builder()
            .rule("s", &[&["greeting"]])
            .rule("greeting", &[&["hi"], &["hello"]])
            .start("s")
            .build();

        let result = map(&g, &[0u8], 0, ProgramBuilder::default());
        assert_eq!(result.unwrap().run(&()), "hi");
    }

    #[test]
    fn empty_codons_single_production() {
        let g = Grammar::builder()
            .rule("s", &[&["hello"]])
            .start("s")
            .build();

        let result = map(&g, &[0u8], 0, ProgramBuilder::default());
        assert_eq!(result.unwrap().run(&()), "hello");
    }
}
