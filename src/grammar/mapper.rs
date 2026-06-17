//! Codon-to-phenotype mapping.

use std::sync::Arc;

use crate::grammar::grammar_def::GrammarDef;
use crate::phenotype::{Event, PhenotypeBuilder};
use vecpool::PoolVec;

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

/// Trait for codon-to-phenotype mapping strategies.
pub trait Mapper {
    /// Maps a codon sequence through a grammar into a phenotype.
    ///
    /// Returns `None` if the mapping fails (e.g., codons exhausted).
    fn map<G: GrammarDef, C: Codon, B: PhenotypeBuilder<G::Terminal>>(
        &self,
        grammar: &G,
        codons: &[C],
        builder: B,
    ) -> Option<B::Output>;
}

impl<M: Mapper> Mapper for Box<M> {
    fn map<G: GrammarDef, C: Codon, B: PhenotypeBuilder<G::Terminal>>(
        &self,
        grammar: &G,
        codons: &[C],
        builder: B,
    ) -> Option<B::Output> {
        (**self).map(grammar, codons, builder)
    }
}

impl<M: Mapper> Mapper for Arc<M> {
    fn map<G: GrammarDef, C: Codon, B: PhenotypeBuilder<G::Terminal>>(
        &self,
        grammar: &G,
        codons: &[C],
        builder: B,
    ) -> Option<B::Output> {
        (**self).map(grammar, codons, builder)
    }
}

/// The standard GE mapping algorithm: stack-based derivation with codon
/// modulo selection, single-production optimization, and wrapping.
#[derive(Debug, Clone, Copy)]
pub struct StandardMapper {
    /// Maximum number of times the codon sequence may wrap before
    /// declaring the mapping invalid.
    pub max_wraps: usize,
}

impl StandardMapper {
    /// Creates a new `StandardMapper` with the given maximum wrap count.
    pub fn new(max_wraps: usize) -> Self {
        Self { max_wraps }
    }
}

impl Mapper for StandardMapper {
    fn map<G: GrammarDef, C: Codon, B: PhenotypeBuilder<G::Terminal>>(
        &self,
        grammar: &G,
        codons: &[C],
        mut builder: B,
    ) -> Option<B::Output> {
        let mut stack: PoolVec<G::Symbol> = PoolVec::new();
        stack.push(grammar.start());
        let mut end_rule_stack: PoolVec<usize> = PoolVec::new();
        let mut codon_idx: usize = 0;

        while let Some(symbol) = stack.pop() {
            if grammar.is_terminal(symbol) {
                builder.push(Event::Terminal(grammar.terminal_value(symbol)));
            } else {
                let n = grammar.num_productions(symbol);
                let choice = if n == 1 {
                    0
                } else {
                    if codons.is_empty() {
                        return None;
                    }
                    let current_wrap = codon_idx / codons.len();
                    if current_wrap > self.max_wraps {
                        return None;
                    }
                    let c = codons[codon_idx % codons.len()].as_usize() % n;
                    codon_idx += 1;
                    c
                };

                builder.push(Event::BeginRule);
                let prod = grammar.production(symbol, choice);
                for &s in prod.iter().rev() {
                    stack.push(s);
                }
                end_rule_stack.push(prod.len());
                continue;
            }

            // After processing a terminal, decrement parent counters.
            while let Some(count) = end_rule_stack.last_mut() {
                *count -= 1;
                if *count == 0 {
                    end_rule_stack.pop();
                    builder.push(Event::EndRule);
                } else {
                    break;
                }
            }
        }

        Some(builder.finish())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::grammar::Grammar;
    use crate::phenotype::{Event, PhenotypeBuilder};

    #[derive(Debug, PartialEq)]
    struct Program(String);

    #[derive(Default)]
    struct ProgramBuilder(String);
    impl PhenotypeBuilder<&'static str> for ProgramBuilder {
        type Output = Program;
        fn push(&mut self, event: Event<&'static str>) {
            if let Event::Terminal(val) = event {
                self.0.push_str(val);
            }
        }
        fn finish(self) -> Program {
            Program(self.0)
        }
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
        let result = StandardMapper::new(0).map(&g, &[1u8], ProgramBuilder::default());
        assert_eq!(result.unwrap().0, "x");
    }

    #[test]
    fn recursive_expansion() {
        let g = expr_grammar();
        let result = StandardMapper::new(0).map(&g, &[0u8, 1, 0, 2], ProgramBuilder::default());
        assert_eq!(result.unwrap().0, "x+1");
    }

    #[test]
    fn wrapping() {
        let g = expr_grammar();
        let result = StandardMapper::new(1).map(&g, &[0u8], ProgramBuilder::default());
        assert_eq!(result, None);
    }

    #[test]
    fn single_production_no_codon() {
        let g = Grammar::builder()
            .rule("s", &[&["greeting"]])
            .rule("greeting", &[&["hi"], &["hello"]])
            .start("s")
            .build();

        let result = StandardMapper::new(0).map(&g, &[0u8], ProgramBuilder::default());
        assert_eq!(result.unwrap().0, "hi");
    }

    #[test]
    fn empty_codons_single_production() {
        let g = Grammar::builder()
            .rule("s", &[&["hello"]])
            .start("s")
            .build();

        let result = StandardMapper::new(0).map(&g, &[0u8], ProgramBuilder::default());
        assert_eq!(result.unwrap().0, "hello");
    }

    #[test]
    fn empty_codons_multi_production_returns_none() {
        let g = Grammar::builder()
            .rule("s", &[&["a"], &["b"]])
            .start("s")
            .build();

        let result = StandardMapper::new(0).map(&g, &[] as &[u8], ProgramBuilder::default());
        assert_eq!(result, None);
    }

    #[test]
    fn large_codon_values() {
        let g = Grammar::builder()
            .rule("s", &[&["a"], &["b"], &["c"]])
            .start("s")
            .build();

        // u8::MAX (255) % 3 == 0 → picks "a"
        let result = StandardMapper::new(0).map(&g, &[u8::MAX], ProgramBuilder::default());
        assert_eq!(result.unwrap().0, "a");

        // 254 % 3 == 2 → picks "c"
        let result = StandardMapper::new(0).map(&g, &[254u8], ProgramBuilder::default());
        assert_eq!(result.unwrap().0, "c");
    }

    #[test]
    fn maps_terminal_list() {
        struct TermList(Vec<&'static str>);

        #[derive(Default)]
        struct TermListBuilder(Vec<&'static str>);
        impl PhenotypeBuilder<&'static str> for TermListBuilder {
            type Output = TermList;
            fn push(&mut self, event: Event<&'static str>) {
                if let Event::Terminal(t) = event {
                    self.0.push(t);
                }
            }
            fn finish(self) -> TermList {
                TermList(self.0)
            }
        }

        let grammar = Grammar::builder()
            .rule("expr", &[&["x"], &["y"]])
            .start("expr")
            .build();

        let result = StandardMapper::new(0).map(&grammar, &[0u8], TermListBuilder::default());
        assert_eq!(result.unwrap().0, vec!["x"]);
    }
}
