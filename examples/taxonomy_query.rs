//! Complex query answering over a small animal taxonomy.
//!
//! Builds a [`FuzzyKg`] by hand, then answers one query of each shape, choosing
//! the [`Truth`](heyting::Truth) algebra that fits the shape. Run:
//!
//! ```text
//! cargo run --release --example taxonomy_query
//! ```
//!
//! Sample output (verbatim):
//!
//! ```text
//! taxonomy: 10 entities, 11 edges
//!
//! 1p   dog is_a ?                      [Godel      ]  -> mammal (1.00)
//! 2p   dog is_a ? is_a ?               [Product    ]  -> animal (1.00)
//! 2i   (dog is_a ?) AND (cat is_a ?)   [Godel      ]  -> mammal (1.00)
//! 2u   (dog is_a ?) OR (sparrow is_a ?) [Godel      ]  -> mammal (1.00), bird (1.00)
//! not  (dog eats ?) AND NOT (cat eats ?) [Lukasiewicz]  -> plant (0.30)
//! ```

use heyting::{answer_query_topk, FuzzyKg, Godel, Lukasiewicz, Product, Query, QueryConfig, Truth};

// Entity ids.
const ANIMAL: usize = 0;
const MAMMAL: usize = 1;
const BIRD: usize = 2;
const FISH: usize = 3;
const DOG: usize = 4;
const CAT: usize = 5;
const SPARROW: usize = 6;
const SHARK: usize = 7;
const MEAT: usize = 8;
const PLANT: usize = 9;
const NAMES: [&str; 10] = [
    "animal", "mammal", "bird", "fish", "dog", "cat", "sparrow", "shark", "meat", "plant",
];

// Relation ids.
const IS_A: usize = 0;
const EATS: usize = 1;

fn build() -> FuzzyKg {
    let mut kg = FuzzyKg::new(10);
    // is_a (child -> parent).
    for (c, p) in [
        (DOG, MAMMAL),
        (CAT, MAMMAL),
        (SPARROW, BIRD),
        (SHARK, FISH),
        (MAMMAL, ANIMAL),
        (BIRD, ANIMAL),
        (FISH, ANIMAL),
    ] {
        kg.add_edge(c, IS_A, p, 1.0);
    }
    // eats (with degrees).
    kg.add_edge(DOG, EATS, MEAT, 1.0);
    kg.add_edge(DOG, EATS, PLANT, 0.3); // dogs nibble grass
    kg.add_edge(CAT, EATS, MEAT, 1.0);
    kg.add_edge(SHARK, EATS, FISH, 0.9);
    kg
}

fn show<T: Truth>(kg: &FuzzyKg, label: &str, algebra: &str, q: &Query) {
    let cfg = QueryConfig::default();
    let top = answer_query_topk::<T>(kg, q, &cfg, 3);
    let answers: Vec<String> = top
        .iter()
        .filter(|(_, s)| *s > 1e-3)
        .map(|(e, s)| format!("{} ({s:.2})", NAMES[*e]))
        .collect();
    println!("{label:<36} [{algebra:<11}]  -> {}", answers.join(", "));
}

fn main() {
    let kg = build();
    println!(
        "taxonomy: {} entities, {} edges\n",
        kg.num_entities(),
        kg.num_edges()
    );

    // 1p: one hop. Gödel (min) preserves the crisp edge degree.
    show::<Godel>(&kg, "1p   dog is_a ?", "Godel", &Query::anchor(DOG, IS_A));

    // 2p: chain. Product propagates magnitude through the two hops.
    show::<Product>(
        &kg,
        "2p   dog is_a ? is_a ?",
        "Product",
        &Query::anchor(DOG, IS_A).then(IS_A),
    );

    // 2i: intersection. Gödel min keeps only what both branches agree on.
    show::<Godel>(
        &kg,
        "2i   (dog is_a ?) AND (cat is_a ?)",
        "Godel",
        &Query::intersection(vec![Query::anchor(DOG, IS_A), Query::anchor(CAT, IS_A)]),
    );

    // 2u: union.
    show::<Godel>(
        &kg,
        "2u   (dog is_a ?) OR (sparrow is_a ?)",
        "Godel",
        &Query::union(vec![Query::anchor(DOG, IS_A), Query::anchor(SPARROW, IS_A)]),
    );

    // negation: Łukasiewicz, the only algebra whose ¬ ranks. "eaten by dog but
    // not by cat" -> plant (dog eats it 0.3, cat doesn't).
    show::<Lukasiewicz>(
        &kg,
        "not  (dog eats ?) AND NOT (cat eats ?)",
        "Lukasiewicz",
        &Query::intersection(vec![
            Query::anchor(DOG, EATS),
            Query::anchor(CAT, EATS).negate(),
        ]),
    );

    // The Heyting implication (Query::implies / Truth::residuum) is also a
    // first-class connective, but a standalone implication ranks vacuously-true
    // entities (premise low) at the top, so it is used as a sub-term rather than
    // a headline query. See the truth-module docs and unit tests.
}
