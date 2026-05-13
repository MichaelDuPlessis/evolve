use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, Token, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

struct Production(Vec<Ident>);

impl Parse for Production {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        bracketed!(content in input);
        let symbols = Punctuated::<Ident, Token![,]>::parse_terminated(&content)?;
        Ok(Production(symbols.into_iter().collect()))
    }
}

struct Rule {
    name: Ident,
    productions: Vec<Production>,
}

impl Parse for Rule {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![=>]>()?;
        let mut productions = vec![input.parse::<Production>()?];
        while input.peek(Token![|]) {
            input.parse::<Token![|]>()?;
            productions.push(input.parse()?);
        }
        input.parse::<Token![;]>()?;
        Ok(Rule { name, productions })
    }
}

struct GrammarInput {
    grammar_vis: syn::Visibility,
    grammar_name: Ident,
    symbol_vis: syn::Visibility,
    symbol_name: Ident,
    start: Ident,
    rules: Vec<Rule>,
}

impl Parse for GrammarInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kw: Ident = input.parse()?;
        if kw != "grammar" {
            return Err(syn::Error::new(kw.span(), "expected `grammar`"));
        }
        let grammar_vis: syn::Visibility = input.parse()?;
        let grammar_name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let kw: Ident = input.parse()?;
        if kw != "symbol" {
            return Err(syn::Error::new(kw.span(), "expected `symbol`"));
        }
        let symbol_vis: syn::Visibility = input.parse()?;
        let symbol_name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let kw: Ident = input.parse()?;
        if kw != "start" {
            return Err(syn::Error::new(kw.span(), "expected `start`"));
        }
        let start: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        let mut rules = Vec::new();
        while !input.is_empty() {
            rules.push(input.parse()?);
        }
        Ok(GrammarInput {
            grammar_vis,
            grammar_name,
            symbol_vis,
            symbol_name,
            start,
            rules,
        })
    }
}

/// Generates a zero-cost grammar with compile-time dispatch.
///
/// Produces a grammar struct, a symbol enum, and a `GrammarDef` implementation
/// with all dispatch resolved via match arms (no allocations, no HashMap lookups).
///
/// # Syntax
///
/// ```ignore
/// grammar! {
///     grammar MyGrammar;
///     symbol MySymbol;
///     start Expr;
///
///     Expr => [Expr, Expr, BinOp] | [Val];
///     BinOp => [Add] | [Sub] | [Mul];
///     Val => [X] | [One];
/// }
/// ```
///
/// - `grammar <name>` — name of the generated struct implementing `GrammarDef`
/// - `symbol <name>` — name of the generated enum with all grammar symbols
/// - `start <rule>` — the start rule
/// - Rules: `<name> => [symbols...] | [symbols...];`
///
/// Symbols appearing on the left side of `=>` are non-terminals.
/// All other symbols are terminals.
#[proc_macro]
pub fn grammar(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as GrammarInput);

    let grammar_name = &input.grammar_name;
    let symbol_name = &input.symbol_name;
    let rule_names: Vec<&Ident> = input.rules.iter().map(|r| &r.name).collect();

    // Collect all symbols (rule names + terminals)
    let mut all_symbols: Vec<Ident> = rule_names.iter().map(|i| (*i).clone()).collect();
    for rule in &input.rules {
        for prod in &rule.productions {
            for sym in &prod.0 {
                if !all_symbols.iter().any(|s| s == sym) {
                    all_symbols.push(sym.clone());
                }
            }
        }
    }

    let start_sym = &input.start;

    // Generate num_productions match arms
    let num_prod_arms: Vec<_> = input
        .rules
        .iter()
        .map(|r| {
            let name = &r.name;
            let count = r.productions.len();
            quote! { #symbol_name::#name => #count, }
        })
        .collect();

    // Generate production match arms
    let mut prod_arms = Vec::new();
    for rule in &input.rules {
        let name = &rule.name;
        for (i, prod) in rule.productions.iter().enumerate() {
            let syms = &prod.0;
            prod_arms.push(quote! {
                (#symbol_name::#name, #i) => &[#(#symbol_name::#syms),*],
            });
        }
    }

    // Generate is_terminal match: rule names are non-terminals
    let non_terminal_names: Vec<_> = rule_names.iter().collect();

    let grammar_vis = &input.grammar_vis;
    let symbol_vis = &input.symbol_vis;

    let expanded = quote! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #symbol_vis enum #symbol_name {
            #(#all_symbols),*
        }

        #grammar_vis struct #grammar_name;

        impl evolve::grammar::grammar_def::GrammarDef for #grammar_name {
            type Symbol = #symbol_name;
            type Terminal = #symbol_name;

            fn start(&self) -> #symbol_name {
                #symbol_name::#start_sym
            }

            fn num_productions(&self, symbol: #symbol_name) -> usize {
                match symbol {
                    #(#num_prod_arms)*
                    _ => 0,
                }
            }

            fn production(&self, symbol: #symbol_name, index: usize) -> &[#symbol_name] {
                match (symbol, index) {
                    #(#prod_arms)*
                    _ => &[],
                }
            }

            fn is_terminal(&self, symbol: #symbol_name) -> bool {
                !matches!(symbol, #(#symbol_name::#non_terminal_names)|*)
            }

            fn terminal_value(&self, symbol: #symbol_name) -> #symbol_name {
                symbol
            }
        }
    };

    TokenStream::from(expanded)
}
