# heyting

Complex logical query answering over knowledge graph embeddings.

`heyting` evaluates multi-hop queries with AND, OR, NOT, and implication by
combining one-hop scorer outputs. The scorer only needs to answer atomic
relation queries as degrees in `[0, 1]`; the same query engine can run over a
point-embedding model, a region model, or a plain in-memory graph.

## Install

```toml
[dependencies]
heyting = "0.15.3"
```

Dual-licensed under MIT or Apache-2.0.

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

`Query` is a tree: anchors at the leaves, connectives above. Trees are the
fragment where this evaluation is exact; cyclic query graphs are out of scope.

## Modules

- **Pruned evaluation** (`prune`): a `CandidateSource` (any serving
  index) proposes per-hop candidates; intersections evaluate most-selective
  branch first with later branches restricted to surviving entities. Results
  are identical to dense evaluation for queries built from hops, AND, and OR;
  only the work changes.
- **Conformal answer sets** (`conformal`): calibrate on `(query, answer)` pairs
  over any scorer, then return
  answer sets containing the true answer with probability `1 − α` for
  exchangeable queries. On FB15k-237 with a trained DistMult (the
  `fb15k237_clqa` example): 84% held-out coverage at the 80% nominal level.
- **Witnesses** (`provenance`): which facts, through which intermediates,
  made an answer true. For the semiring algebras (`Godel`, `Viterbi`),
  `explain_answer` returns one derivation whose degree equals the engine degree.
- **Abduction** (`abduce`): the reverse question. Given observed entities,
  recover the template hypothesis (one-hop atoms and their pairwise
  conjunctions) that best explains them, scored by fuzzy Jaccard overlap.
- **Numeric literals** (`Query::given`): encode "attribute in `[lo, hi]`" as
  a degree vector and conjoin it with relation hops.
- **Temporal scoping** (`temporal`): facts carry validity intervals; a
  `TimeWindow` (before/after/between, or relative to another fact) registers
  as a virtual relation id, so time-scoped hops compose through the ordinary
  connectives; planning, pruning, conformal, and witnesses all apply. For
  event KGs with discrete timestamps, `TimeSet` (a bitset closed under
  union, intersection, and complement) carries the non-contiguous sets that
  temporal operators produce; a not-during hop is one virtual relation.
- **Standard evaluation** (`eval`): the easy/hard answer split with filtered
  metrics, as in the Query2Box/BetaE protocol.

## Algebras

The algebra is chosen as a type parameter:

| Algebra | conjunction | disjunction | negation |
|---|---|---|---|
| `Godel` | `min(a, b)` | `max(a, b)` | crisp |
| `Product` | `a * b` | `a + b - ab` | crisp |
| `Lukasiewicz` | `max(0, a + b - 1)` | `min(1, a + b)` | `1 - a` |
| `Viterbi` | `a * b` | `max(a, b)` | crisp |

All algebras implement implication through the residuum `a -> b`, with property
tests for the adjunction `and(a, c) <= b` iff `c <= residuum(a, b)`. The shared
t-norm and residuum formulas come from `tnorms`; this crate adds the typed
`Truth` trait, query evaluation, and provenance constraints. `Godel` and
`Viterbi` are also the algebras used for exact witness extraction, because
their disjunction selects a single best derivation.

## Adapters

- Feature `tranz`: `adapters::PointModel` wraps a trained `tranz::Scorer`
  (`TransE`/`RotatE`/`ComplEx`/`DistMult`) as an `AtomicScorer`, with a
  sigmoid temperature for calibrated degrees. `adapters::TemporalPointModel`
  does the same for a trained `tranz::temporal::TComplEx`, with
  `TimeSet`-scoped hops registered as virtual relations.
- Feature `subsume`: `adapters::BoxModel` scores Query2Box-style over trained
  box embeddings, and `BoxModel::materialize_explained` runs the query in the
  geometry itself: exact box intersections, DNF unions (a single box cannot
  represent a union unless dimension scales with entity count), no negation.
  Returns the answer region and its composition tree.
- `adapters::FaithfulBoxModel` is dependency-free and scores faithful EL-style
  concept boxes by graded inclusion (`C ⊑ D`), for ontology-shaped query
  answering over region embeddings.

## Examples

```sh
cargo run --release --features tranz --example fb15k237_clqa
cargo run --release --features tranz --example icews14_temporal_clqa
```

`fb15k237_clqa` trains a 1-hop model with the `tranz` CLI, composes queries in
`heyting`, scores with the easy/hard protocol, and prints a witness plus
conformal coverage. `icews14_temporal_clqa` is the temporal counterpart on
ICEWS14.

## Relationship to tranz

`heyting` generalizes `tranz::query` (CQD-Beam over point embeddings,
Arakelyan et al. 2021): implement `AtomicScorer` for any point or region
model and the same connectives answer complex queries over it.

## References

- Hájek. *Metamathematics of Fuzzy Logic*. Kluwer, 1998.
- Green, Karvounarakis, Tannen. Provenance semirings. PODS 2007.
- Goodman. Semiring parsing. Computational Linguistics 25(4), 1999.
- Ren, Hu, Leskovec. Query2box. ICLR 2020. arXiv:2002.05969.
- Ren, Leskovec. Beta embeddings for multi-hop logical reasoning in knowledge
  graphs. NeurIPS 2020. arXiv:2010.11465.
- Arakelyan, Daza, Minervini, Cochez. Complex query answering with neural link
  predictors. ICLR 2021. arXiv:2011.03459.
- Yin, Wang, Song. Rethinking complex queries on knowledge graphs with neural
  link predictors. ICLR 2024. arXiv:2304.07063.
- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World*. Springer,
  2005.
- Angelopoulos, Bates. A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification. arXiv:2107.07511.
- Bai et al. Advancing abductive reasoning in knowledge graphs through complex
  logical hypothesis generation. ACL 2024.
- Lacroix, Obozinski, Usunier. Tensor decompositions for temporal knowledge
  base completion. ICLR 2020. arXiv:2004.04926.
- Lin et al. TFLEX. NeurIPS 2023. arXiv:2205.14307.

Short implementation notes for these references are in [docs/papers.md](docs/papers.md).
