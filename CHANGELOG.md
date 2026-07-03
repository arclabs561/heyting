# Changelog

## [0.4.0] - 2026-07-03

### Added

- `prune` module: `CandidateSource` (the serving-index seam) and
  `answer_query_topk_pruned`, sparse evaluation of the EPFO fragment that
  scores only proposed candidates; negation/implication queries fall back
  to dense evaluation. `FuzzyKg` is its own exact candidate source.
- `AtomicScorer::project_subset` (defaulted): score a candidate subset;
  override it in embedding scorers to realize the pruning speedup.
- `adapters::BoxModel` (feature `subsume`): Query2Box-style scoring over
  trained box embeddings (entities as points, relations as
  translation + offset pairs, degrees `exp(-distance / temperature)`).

## [0.3.0] - 2026-07-03

### Added

- `eval` module: the standard easy/hard answer split (`split_answers`,
  `crisp_answers`) and filtered ranking metrics over hard answers
  (`hard_answer_metrics`, `QueryMetrics`). The module doc maps the standard
  query-shape taxonomy (`1p` through `pni`) to `Query` constructors.

### Fixed

- docs.rs rendering of the `adapters` module doc link on default-feature
  builds (unresolved `AtomicScorer` link when the `tranz` feature is off).

## [0.2.0] - 2026-06-29

### Added

- `adapters::PointModel` (feature `tranz`): wraps any `tranz::Scorer`
  (TransE/RotatE/ComplEx/DistMult) as an `AtomicScorer`, mapping link-prediction
  scores to `[0, 1]` membership via `sigmoid(-score)`. The query engine now runs
  over trained point embeddings, not just `FuzzyKg`.

## [0.1.0] - 2026-06-28

Initial release.

### Added

- `Truth` trait: residuated lattices of truth degrees, with the `Godel`,
  `Product`, and `Lukasiewicz` algebras. The residuum `a → b` (the Heyting
  implication) is a first-class method, and the residuation law
  `a ⊗ (a → b) ≤ b` is property-tested for all three.
- `Query` DAG with `anchor`, `then` (existential projection), `intersection`,
  `union`, `negate`, and `implies` connectives.
- `AtomicScorer` trait: the geometry-generic seam
  (`project(anchor, relation) -> [0, 1]` membership degrees).
- `answer_query` / `answer_query_topk`, generic over the truth algebra.
- `FuzzyKg`: a reference in-memory fuzzy knowledge graph.
- `taxonomy_query` example.
