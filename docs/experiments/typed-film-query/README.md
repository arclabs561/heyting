# Profession-constrained two-hop lookup on FB15k-237

For director D and profession P, retrieve each actor X for whom the graph contains a film F directed by D, an acting credit for X in F, and a profession P for X. A reviewer can inspect all three source facts. A shared profession alone cannot justify an answer.

We evaluated all 274 directors with a training-set directing edge, crossed with the eight profession IDs belonging to the largest numbers of distinct training-set film actors. This produced 2,192 queries, including ones yielding no observed answer. The query families and intermediate-film ranking use training edges exclusively. Each answer counts once per query, regardless of how many films supply evidence.

| Film expansion policy | Supported query–answer pairs | Queries emptied by pruning |
|---|---:|---:|
| All observed films | 7,795 | 0 |
| Eight largest observed casts | 7,605 | 0 |
| Three largest observed casts | 6,182 | 32 |
| Largest observed cast | 3,745 | 198 |

Only 1,505 queries have a complete training witness. Cutting expansion to eight films still loses 190 witnessed answers even though it preserves at least one answer to every nonempty query. Holding the query list fixed, adding validation facts supplies 1,278 additional answers, and adding test facts supplies another 2,028. Across all three snapshots there are 11,101 supported answers. These are positive-evidence counts. A missing witness calls for an unknown result. In this test a zero Heyting degree means unsupported under the current graph. No explicit negative evidence, neural completion, hierarchy inference, calibrated likelihood, or official BetaE benchmark is involved.

The Rust verifier builds a published Heyting 0.17.0 CPU FuzzyKg from the three selected training relations. It reverses the actor-to-film and person-to-profession edges in the index, gives observed edges weight 1, runs exact Gödel projection and intersection, and compares the full set of positive-degree entity IDs for each query against an independent Python set join. The film-cap ablations use Python and rank by observed cast size with lexical film-ID ties. They do not benchmark Heyting's internal beam implementation. Mapping a profession value to an ontology supertype would be a distinct task.

## Provenance

The [original Box Coref experiment](https://github.com/arclabs561/box-coref/tree/bde540d95cfd3e7d9ad905065a1dcfd3bc0c6157/experiments/adjacent/query) established these receipts using published Heyting 0.17.0. The upstream corpus commit and individual split digests identify the input independently of that historical checkout. This repository maintains the verifier and joins. Rebuild fresh exports in scratch and keep the graph triples outside version control.

## Reproduce

Use the three pinned raw split files. The Python harness verifies their SHA-256 digests. The optional Rust cross-check consumes a local export and returns nonzero if any query differs.

```sh
mkdir -p /tmp/fb15k-237
for split in train valid test; do
  curl -fsSLo "/tmp/fb15k-237/$split.txt" \
    "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/2e440e0f9c687314d5ff67ead68ce985dc446e3a/data/FB15k-237/$split.txt"
done
python3 scripts/typed_film_query.py /tmp/fb15k-237 --verification /tmp/film-witness > reproduced.json
diff -u docs/experiments/typed-film-query/result.json reproduced.json
cargo run --release --example typed_film_query -- /tmp/film-witness
```

The last command exercises this checkout. To verify the published 0.17.0 crate against the same oracle, run this from the repository root.

```sh
verifier=$(mktemp -d)
mkdir -p "$verifier/src"
cp examples/typed_film_query.rs "$verifier/src/main.rs"
cat > "$verifier/Cargo.toml" <<'TOML'
[package]
name = "verify-typed-film-query"
version = "0.1.0"
edition = "2021"
[dependencies]
heyting = { version = "=0.17.0", default-features = false }
TOML
cargo run --release --manifest-path "$verifier/Cargo.toml" -- /tmp/film-witness
```

Both runs should report 2,192 queries and 7,795 supported answers. To falsify the check, make a temporary copy of the verifier input and append a spurious entity ID to an expected answer row. Running the verifier on the altered input must return nonzero.

[Dataset snapshot](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/2e440e0f9c687314d5ff67ead68ce985dc446e3a/data/FB15k-237) · [Heyting 0.17.0](https://docs.rs/heyting/0.17.0/heyting/)
