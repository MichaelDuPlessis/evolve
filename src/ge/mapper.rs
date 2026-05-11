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
///
/// # Examples
///
/// ```
/// use evolve::ge::grammar::Grammar;
/// use evolve::ge::mapper::map;
/// use evolve::ge::phenotype::StringBuilder;
///
/// let grammar = Grammar::builder()
///     .rule("expr", &[&["x"], &["y"]])
///     .start("expr")
///     .build();
///
/// // Codon 0 → first production "x", codon 1 → second production "y"
/// let result = map(&grammar, &[0u8], 0, StringBuilder::new());
/// assert_eq!(result, Some("x".to_string()));
///
/// let result = map(&grammar, &[1u8], 0, StringBuilder::new());
/// assert_eq!(result, Some("y".to_string()));
/// ```
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
    use crate::ge::grammar::Grammar;
    use crate::ge::phenotype::StringBuilder;

    fn expr_grammar() -> Grammar {
        Grammar::builder()
            .rule("expr", &[&["expr", "op", "expr"], &["x"], &["1"]])
            .rule("op", &[&["+"], &["-"]])
            .start("expr")
            .build()
    }

    #[test]
    fn simple_terminal() {
        let g = expr_grammar();
        // codon 1 % 3 = 1 → "x"
        let result = map(&g, &[1u8], 0, StringBuilder::new());
        assert_eq!(result, Some("x".to_string()));
    }

    #[test]
    fn recursive_expansion() {
        let g = expr_grammar();
        // codon 0 % 3 = 0 → expr op expr
        // codon 1 % 3 = 1 → "x" (left expr)
        // codon 0 % 2 = 0 → "+" (op)
        // codon 2 % 3 = 2 → "1" (right expr)
        let result = map(&g, &[0u8, 1, 0, 2], 0, StringBuilder::new());
        assert_eq!(result, Some("x+1".to_string()));
    }

    #[test]
    fn wrapping() {
        let g = expr_grammar();
        // With only 1 codon [0], it will wrap. codon 0 % 3 = 0 → recursive.
        // This will keep recursing until wraps exceeded.
        let result = map(&g, &[0u8], 1, StringBuilder::new());
        // Should return None because infinite recursion exceeds wraps.
        assert_eq!(result, None);
    }

    #[test]
    fn single_production_no_codon() {
        // Single-production rules don't consume codons.
        let g = Grammar::builder()
            .rule("s", &[&["greeting"]])
            .rule("greeting", &[&["hi"], &["hello"]])
            .start("s")
            .build();

        // "s" has 1 production → no codon consumed.
        // "greeting" has 2 → codon 0 % 2 = 0 → "hi"
        let result = map(&g, &[0u8], 0, StringBuilder::new());
        assert_eq!(result, Some("hi".to_string()));
    }

    #[test]
    fn empty_codons_single_production() {
        // If all rules have single productions, empty codons should work.
        let g = Grammar::builder()
            .rule("s", &[&["hello"]])
            .start("s")
            .build();

        let result = map(&g, &[0u8], 0, StringBuilder::new());
        assert_eq!(result, Some("hello".to_string()));
    }
}
