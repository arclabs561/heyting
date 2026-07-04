# heyting

Complex logical query answering over knowledge graph embeddings.

```toml
[dependencies]
heyting = "0.5"
```

Dual-licensed under MIT or Apache-2.0.

Answers multi-hop queries with AND, OR, NOT, and implication by evaluating a
query as a computation DAG over atomic one-hop lookups and combining the
membership degrees with fuzzy-logic connectives. The engine is geometry-generic:
anything that answers a one-hop query as `[0, 1]` degrees plugs in through the
`AtomicScorer` trait, so the same code runs over point embeddings, region
embeddings, or a plain in-memory graph.

## The logic is a typed algebra

Query degrees compose in a *residuated lattice*, chosen as a type parameter:

| Algebra | conjunction `a ⊗ b` | implication `a → b` | negation `¬a` |
|---|---|---|---|
| `Godel` | `min(a, b)` | `1 if a ≤ b else b` | crisp |
| `Product` | `a · b` | `1 if a ≤ b else b/a` | crisp |
| `Lukasiewicz` | `max(0, a+b−1)` | `min(1, 1−a+b)` | `1 − a` |

The lattice's residuum `a → b` is the Heyting implication the crate is named for.
The negation each algebra induces is the load-bearing choice: `Lukasiewicz` gives
the soft `¬a = 1 − a` that ranking wants, while `Godel`/`Product` give the genuine
intuitionistic pseudo-complement (crisp). That region negation is only a
pseudo-complement is exactly why geometric query negation is approximate. The
residuation law `a ⊗ (a → b) ≤ b` is property-tested for all three.

## Example

```rust
use heyting::{answer_query_topk, FuzzyKg, Godel, Query, QueryConfig};

// 0=animal 1=mammal 2=dog 3=cat; relation 0 = is_a.
let mut kg = FuzzyKg::new(4);
kg.add_edge(2, 0, 1, 1.0); // dog is_a mammal
kg.add_edge(3, 0, 1, 1.0); // cat is_a mammal
kg.add_edge(1, 0, 0, 1.0); // mammal is_a animal

// (dog is_a ?) AND (cat is_a ?) -> mammal.
let q = Query::intersection(vec![Query::anchor(2, 0), Query::anchor(3, 0)]);
let top = answer_query_topk::<Godel>(&kg, &q, &QueryConfig::default(), 1);
assert_eq!(top[0].0, 1);
```

`cargo run --release --example taxonomy_query` answers one query of each shape:

```text
1p   dog is_a ?                      [Godel      ]  -> mammal (1.00)
2p   dog is_a ? is_a ?               [Product    ]  -> animal (1.00)
2i   (dog is_a ?) AND (cat is_a ?)   [Godel      ]  -> mammal (1.00)
2u   (dog is_a ?) OR (sparrow is_a ?) [Godel      ]  -> mammal (1.00), bird (1.00)
not  (dog eats ?) AND NOT (cat eats ?) [Lukasiewicz]  -> plant (0.30)
```

## Relationship to tranz

`heyting` is the geometry-generic counterpart of `tranz::query` (CQD-Beam over
point embeddings, Arakelyan et al. 2021): implement `AtomicScorer` for any point
or region model and the same connectives answer complex queries over it.

The optional `tranz` feature provides `adapters::PointModel`, which wraps a
trained `tranz::Scorer` (`TransE`/`RotatE`/`ComplEx`/`DistMult`) as an
`AtomicScorer`, mapping its distance scores to `[0, 1]` degrees so the query
engine runs over point embeddings directly.
