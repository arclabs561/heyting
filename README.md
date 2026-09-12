# heyting

Complex logical query answering over knowledge graph embeddings.

`heyting` evaluates multi-hop queries with AND, OR, NOT, and implication by
combining one-hop scorer outputs. The scorer only needs to answer atomic
relation queries as degrees in `[0, 1]`; the same query engine can run over a
point-embedding model, a region model, or a plain in-memory graph.

## Install

```toml
[dependencies]
heyting = "0.17.0"
```

Dual-licensed under MIT or Apache-2.0.

## Example

```rust
use heyting::{answer_query_topk, FuzzyKg, Godel, Query, QueryConfig};

fn main() {
    // 0=animal 1=mammal 2=dog 3=cat; relation 0 = is_a.
    let mut kg = FuzzyKg::new(4);
    kg.add_edge(2, 0, 1, 1.0); // dog is_a mammal
    kg.add_edge(3, 0, 1, 1.0); // cat is_a mammal
    kg.add_edge(1, 0, 0, 1.0); // mammal is_a animal

    // (dog is_a ?) AND (cat is_a ?) -> mammal.
    let q = Query::intersection(vec![Query::anchor(2, 0), Query::anchor(3, 0)]);
    let top = answer_query_topk::<Godel>(&kg, &q, &QueryConfig::default(), 1);
    assert_eq!(top[0].0, 1);
    println!("mammal: {:.1}", top[0].1);
}
```

`Query` is a tree: anchors at the leaves, connectives above.
`QueryConfig::exact()` disables beam truncation; the default expands at most
128 intermediate candidates per projection. The query language is tree-form:
cyclic graphs and joins that share an intermediate are out of scope.

```text
mammal: 1.0
```

## Modules

- **Pruned evaluation** (`prune`): a `CandidateSource` proposes candidates
  for each atomic hop; intersections evaluate the most selective branch first
  and restrict later branches to surviving entities. For positive tree-form
  queries, results match dense evaluation when the source covers every
  nonzero hop result (`FuzzyKg` does); missing candidates reduce recall.
  Queries with negation or implication use dense evaluation, since entities
  outside a candidate set can become answers.
- **Conformal answer sets** (`conformal`): calibrate on held-out `(query, answer)`
  pairs. With a scorer fixed independently of calibration and exchangeable
  future pairs, the answer set contains its designated true answer with
  marginal coverage at least `1 − α`. This does not promise coverage for each
  query, relation, or every true answer of a multi-answer query.
  The `fb15k237_clqa` example records one trained-DistMult run with 80%
  held-out coverage at the 80% nominal level.
- **Witnesses** (`provenance`): which facts and intermediates support an
  answer. For `Godel` and `Viterbi`, `explain_answer` returns one best
  derivation whose degree matches the engine under the same beam setting.
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

- Feature [`tranz`](https://github.com/arclabs561/tranz):
  `adapters::PointModel` wraps a trained `tranz::Scorer`
  (`TransE`/`RotatE`/`ComplEx`/`DistMult`) as an `AtomicScorer`, mapping its
  scores to `[0, 1]` with a monotone sigmoid. Temperature changes the spread
  without changing ranking; fit and validate it separately if calibrated
  degrees matter.
  `adapters::TemporalPointModel` does the same for trained
  `tranz::temporal::TComplEx`, registering `TimeSet`-scoped hops as virtual
  relations.
- Feature [`subsume`](https://github.com/arclabs561/subsume):
  `adapters::BoxModel` scores Query2Box-style over trained box embeddings.
  `BoxModel::materialize_explained` composes positive queries geometrically as
  a disjunctive normal form of boxes, returning the answer region and its
  composition tree; negation and implication have no box materialization.
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
Arakelyan et al. 2021): implement `AtomicScorer` for a point, region, or
other one-hop scorer to use the same tree-form connectives.

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
- Gregucci, Xiong, Hernandez, Loconte, Minervini, Staab, Vergari. Is complex
  query answering really complex? ICML 2025. arXiv:2410.12537.
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
