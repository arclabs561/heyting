# Changelog

## [0.15.2] - 2026-07-07

### Changed

- The `Truth` algebra implementations now delegate their standard t-norm,
  t-conorm, and residuum formulas to `tnorms`.

## [0.15.1] - 2026-07-07

### Changed

- The optional `subsume` feature now depends on `subsume` 0.17, matching the
  geometry crate's post-cleanup surface.

## [0.15.0] - 2026-07-06

### Added

- `conformal::answer_set_from_scored_pool`: sparse candidate-pool conformal
  answer sets for readouts that only score retrieved candidates.

### Changed

- `adapters::PointModel` and `adapters::TemporalPointModel` now override
  `AtomicScorer::project_subset`, so candidate-pruned query evaluation can
  score supplied candidates directly for `tranz` backends.

## [0.14.0] - 2026-07-06

### Added

- `adapters::FaithfulBoxModel`: dependency-free faithful-EL box subsumption
  scorer, where entities are boxes and `project(anchor, relation)` returns
  graded box-inclusion degrees.
- `conformal::calibrate_scores` and
  `conformal::answer_set_from_degrees`: scorer-agnostic conformal core for
  readouts that do not fit the atomic `AtomicScorer` seam. The existing
  `calibrate` and `answer_set` APIs now delegate to this core.
- `el_clqa_gated_conformal` example: conformal LCA answer sets over
  `subsume`'s gated box readout.

### Fixed

- Cross-feature intra-doc links in the adapters module now build under
  feature subsets such as `--features subsume`.

## [0.13.0] - 2026-07-04

### Added

- `TemporalPointModel::when`: time projection for a concrete fact pair
  (degrees over the timestamp axis for "when does `(h, r, t)` hold"),
  TFLEX's time-projection operator restricted to concrete anchors, and
  `TimeSet::from_degrees` to carry the answer back into set logic, so a
  PREDICTED event time can anchor After/Before/Between hops.
- `betae_fb15k237` example: the KGReasoning/BetaE FB15k-237 query files
  evaluated file-exactly across all 14 query types, with negation as
  top-k exclusion (soft `1 − sigmoid` negation over uncalibrated degrees
  measures at the random floor). `scripts/fetch_betae_fb15k237.sh`
  fetches and converts the pickles.
- `el_clqa` / `el_clqa_conformal` examples: graded EL++ subsumption CLQA
  over region embeddings (concept boxes, graded `C ⊑ t` as the atomic
  projection), with conformal calibration over a trained model.
- ICEWS05-15 support in the temporal example (`ICEWS_DATA`/`ICEWS_EMB`
  env overrides, `scripts/fetch_icews0515.sh`) and `examples/README.md`
  with captured outputs.

### Changed

- `TemporalPointModel` hops fold their existential minimum through
  `tranz::temporal::TemporalScorer::score_all_tails_over` (batched and
  rayon-parallel), removing the per-timestamp scoring loop that made
  not-during hops on long axes the harness bottleneck (ICEWS05-15
  harness: ~53 min → 140 s). Requires `tranz >= 0.7.2`.

## [0.12.0] - 2026-07-04

### Added

- `TimeSet`: bitset carrier for discrete timestamp axes, closed under
  union, intersection, and complement (the non-contiguous sets temporal
  operators produce, which `TimeWindow` intervals cannot represent), with
  the window vocabulary as constructors and TFLEX's set-level operators
  (`after_all`, `before_all`, `between_all`; Lin et al., NeurIPS 2023).
- `adapters::TemporalPointModel` (feature `tranz`): wraps a trained
  `tranz::temporal::TemporalScorer` (TComplEx) with `TimeSet`-scoped hops
  registered as virtual relations; hop degrees are existential over the
  set, with the same sigmoid-temperature map as `PointModel`.
- `icews14_temporal_clqa` example: windowed and non-reducible temporal
  query types against an exact `TemporalKg` oracle, witness, and conformal
  coverage; `scripts/fetch_icews14.sh` fetches the dataset.
- `docs/papers.md`: mechanism-level summaries of every referenced paper,
  linked per entry from the README reference list.

### Removed

- `truth::Idempotent`. The marker was purely descriptive: no API was
  bounded by it (`explain_answer` requires the weaker `SelectiveOr`, which
  is the actual witness-exactness condition), and the both-ops-idempotent
  fact it encoded is now documented on `Godel`, the only lawful
  implementor. Migration: replace `Idempotent` bounds with `SelectiveOr`.

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
