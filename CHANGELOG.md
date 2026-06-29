# Changelog

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
