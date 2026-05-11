//! Grammatical Evolution (GE) module.
//!
//! Provides grammar-based mapping of integer codon sequences to phenotypes,
//! enabling evolution of programs and expressions via the existing evolutionary
//! algorithm runner.

pub mod fitness;
pub mod grammar;
pub mod grammar_def;
pub mod mapper;
pub mod phenotype;
