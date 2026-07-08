# Stability

`heyting` is pre-1.0. It follows semver with the usual `0.x` rule:
patch releases should not intentionally break public APIs, while minor releases
may make breaking changes and document them in `CHANGELOG.md`.

## Core Surface

The core surface is intended to stay patch-compatible within a minor release:

- `Query` and `QueryConfig`
- `AtomicScorer`
- `Truth`, `Godel`, `Product`, `Lukasiewicz`, and `Viterbi`
- `CandidateSource`
- `answer_query`, `answer_query_topk`, and `answer_query_topk_pruned`

Breaking changes to this surface should wait for a minor version bump before
1.0.

## Moving Surface

Feature-gated adapters and experiment-facing modules may change in minor
releases as their upstream crates and evaluation protocols change:

- `adapters`
- `conformal`
- `provenance`
- `abduce`
- `temporal`

The `Query` type remains a tree-form query language. Cyclic query graphs and
multi-anchor joins are outside the current API boundary.
