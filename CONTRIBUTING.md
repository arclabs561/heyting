# Contributing to heyting

Thanks for your interest. heyting answers complex logical queries over knowledge
graph embeddings, evaluating a query DAG in a chosen residuated-lattice algebra.

## Before you start

For non-trivial work (new APIs, features, large refactors), open an issue first
to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable, MSRV `1.89`. Use `rustup` to manage.

```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Style

- Direct, lowercase prose in commits. No marketing words. No em-dashes in prose.
- Commit messages: `heyting: short lowercase description`. One commit per logical change.
- New truth algebras must satisfy the residuated-lattice laws and be covered by
  the property tests in `src/truth.rs`.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree
your contributions are licensed under both.
