# Changelog

## [0.11.0] - 2026-07-03

### Added

- `abduce` module: template-based abductive hypothesis generation — given
  observed entities, recover the atom or conjunction that best explains
  them (fuzzy-Jaccard scored). The abductive twin of `provenance`.

### Changed

- README documents the full current surface (algebras and the semiring
  line, planning, conformal, provenance, literals, temporal, adapters).

## [0.10.0] - 2026-07-03

### Added

- `Viterbi` truth algebra (product t-norm, `max` t-conorm): the classical
  best-derivation semiring; chains propagate magnitude and stay witnessable.
- `truth::SelectiveOr` marker (selective `⊕` = `max`): the actual soundness
  condition for witnesses; `Idempotent` is now its subtrait, and
  `explain_answer` is bounded by `SelectiveOr` (Gödel or Viterbi).
- Witnesses cover negation and implication via `Witness::Refutation` and
  `Witness::Implied` markers; `WitnessError::UnsupportedConnective` removed.
- `PointModel::with_temperature`: sigmoid temperature (rankings invariant;
  fixes saturated degrees for conformal thresholds). `PointModel` is now a
  named struct built with `PointModel::new`.
- `TemporalKg::fact_interval` + `windowed_after_fact` / `windowed_before_fact`
  / `windowed_during_fact`: event-relative time windows.

## [0.9.0] - 2026-07-03

### Added

- `provenance` module: bottleneck witness trees (`explain_answer`) — the
  why-provenance of a degree-path answer, grounded in provenance semirings
  (Green et al., PODS 2007). Witness degree equals the engine degree exactly.
- `truth::Idempotent` marker trait: encodes which algebras are `(min, max)`
  semirings (Gödel; the classical only-idempotent-t-norm-is-min result), the
  bound `explain_answer` requires for soundness.
- `fb15k237_clqa` example (feature `tranz`): end-to-end CLQA evaluation on
  FB15k-237 with a tranz-trained DistMult — composition, easy/hard metrics,
  a witness, and conformal coverage in one run — plus a fetch script.

### Changed

- `adapters` split into `point` / `box_model` / `box_dnf` files; public
  paths unchanged.

## [0.8.0] - 2026-07-03

### Added

- Geometric execution mode (feature `subsume`): `BoxModel::materialize` /
  `materialize_explained` compose the query's boxes themselves, returning a
  `BoxDnf` answer region (exact intersections, DNF unions, no negation) and
  an `Explanation` composition tree with per-node log-volumes.

### Changed

- The `subsume` feature now depends on `subsume` 0.14.1 with default
  features off, dropping its ndarray/serde_json/lattix subtree.

## [0.7.0] - 2026-07-03

### Added

- `temporal` module: `TemporalKg` (facts with validity intervals) and
  `TimeWindow` (before / after / between). A window registered on the graph
  becomes a virtual relation id, so time-scoped hops compose through the
  ordinary query connectives, and planning, pruning, and conformal
  calibration apply to temporal queries unchanged. Ships with the
  `temporal_query` example (the two-non-adjacent-terms query).

## [0.6.0] - 2026-07-03

### Added

- `Query::Given` / `Query::given`: precomputed membership leaves, the
  numeric-literal-constraint pattern (encode "attribute in [lo, hi]" as a
  degree vector and conjoin it with relation hops).

### Changed

- The pruned evaluator plans intersections: branches evaluate most-selective
  first (candidate-count estimates) and later branches are restricted to the
  surviving support, cutting scoring work with identical results.

## [0.5.0] - 2026-07-03

### Added

- `conformal` module: split conformal prediction over query answers
  (`calibrate`, `answer_set`, `empirical_coverage`). Calibrating on
  `(query, true answer)` pairs yields answer sets that contain the true
  answer with probability at least `1 - alpha` for exchangeable queries,
  for any scorer and truth algebra.

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
