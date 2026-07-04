# heyting

Complex logical query answering over knowledge graph embeddings.

```toml
[dependencies]
heyting = "0.11"
```

Dual-licensed under MIT or Apache-2.0.

Answers multi-hop queries with AND, OR, NOT, and implication by evaluating a
query as a computation DAG over atomic one-hop lookups and combining the
membership degrees with fuzzy-logic connectives. The engine is generic over
the scorer: anything that answers a one-hop query as `[0, 1]` degrees plugs in
through the `AtomicScorer` trait, so the same code runs over point embeddings,
region embeddings, or a plain in-memory graph.

## The algebras

Query degrees compose in a *residuated lattice*, the textbook algebra of
fuzzy logic (Hájek, *Metamathematics of Fuzzy Logic*, 1998): a conjunction
`⊗` paired with an implication `→` such that `a ⊗ b ≤ c` exactly when
`a ≤ b → c`. The algebra is chosen as a type parameter:

| Algebra | conjunction `a ⊗ b` | disjunction `a ⊕ b` | negation `¬a` |
|---|---|---|---|
| `Godel` | `min(a, b)` | `max(a, b)` | crisp |
| `Product` | `a · b` | `a + b − ab` | crisp |
| `Lukasiewicz` | `max(0, a+b−1)` | `min(1, a+b)` | `1 − a` |
| `Viterbi` | `a · b` | `max(a, b)` | crisp |

The residuum `a → b` is the Heyting implication the crate is named for; the
residuation law `a ⊗ (a → b) ≤ b` is property-tested for every algebra. Two
columns carry consequences. Negation: `Lukasiewicz` gives the soft
`¬a = 1 − a` that ranking wants; the others are crisp, all-or-nothing
(`¬a = 1` if `a = 0`, else `0`). Disjunction: `Godel` and `Viterbi` pair
their conjunction with `max`, making them commutative semirings (Green et
al., PODS 2007), which is what the witness machinery below requires.
`Product` and `Lukasiewicz` are not semirings (`⊗` does not distribute over
`⊕`); their degrees aggregate evidence across derivations instead.

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

## Modules

- **Planned, pruned evaluation** (`prune`): a `CandidateSource` (any serving
  index) proposes per-hop candidates; intersections evaluate most-selective
  branch first with later branches restricted to surviving entities. Results
  are identical to dense evaluation for queries built from hops, AND, and OR;
  only the work changes.
- **Answer sets with coverage guarantees** (`conformal`): split conformal
  prediction (distribution-free calibration, Vovk et al.) over any scorer:
  calibrate on `(query, answer)` pairs, get
  answer sets containing the true answer with probability `1 − α` for
  exchangeable queries. On FB15k-237 with a trained DistMult (the
  `fb15k237_clqa` example): 84% held-out coverage at the 80% nominal level.
- **Witnesses** (`provenance`): which facts, through which intermediates,
  made an answer true ("why-provenance" in the database-theory sense). For
  the semiring algebras (`Godel`, `Viterbi`) every answer's degree is
  realized by one best derivation, and `explain_answer` returns it with
  witness degree exactly equal to the engine degree. Negation is witnessed
  as a recorded refutation.
- **Numeric literals** (`Query::given`): encode "attribute in `[lo, hi]`" as
  a degree vector and conjoin it with relation hops.
- **Temporal scoping** (`temporal`): facts carry validity intervals; a
  `TimeWindow` (before/after/between, or relative to another fact) registers
  as a virtual relation id, so time-scoped hops compose through the ordinary
  connectives; planning, pruning, conformal, and witnesses all apply.
- **Standard evaluation** (`eval`): the easy/hard answer split with filtered
  metrics, as in the Query2Box/BetaE protocol.

`Query` is a tree: anchors at the leaves, connectives above. Trees are the
fragment where this evaluation is exact ("tree-form" queries in the CLQA
literature); cyclic query graphs are out of scope.

## Adapters

- Feature `tranz`: `adapters::PointModel` wraps a trained `tranz::Scorer`
  (`TransE`/`RotatE`/`ComplEx`/`DistMult`) as an `AtomicScorer`, with a
  sigmoid temperature for calibrated degrees.
- Feature `subsume`: `adapters::BoxModel` scores Query2Box-style over trained
  box embeddings, and `BoxModel::materialize_explained` runs the query in the
  geometry itself: exact box intersections, DNF unions (a single box cannot
  represent a union unless dimension scales with entity count), no negation.
  Returns the answer region and its composition tree.

`cargo run --release --features tranz --example fb15k237_clqa` runs the
FB15k-237 example end to end: train a 1p model with the tranz CLI, compose queries here,
score with the easy/hard protocol, print a witness and conformal coverage.

## Relationship to tranz

`heyting` generalizes `tranz::query` (CQD-Beam over point embeddings,
Arakelyan et al. 2021): implement `AtomicScorer` for any point or region
model and the same connectives answer complex queries over it.
