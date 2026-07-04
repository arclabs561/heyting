# heyting examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs. The two dataset examples are data-gated: without
their data they print fetch/train instructions and exit 0.

## Which example should I run?

| I want to... | Example |
|---|---|
| See the connectives on a ten-entity toy taxonomy | `taxonomy_query` |
| See interval-validity temporal hops, no training | `temporal_query` |
| Run the full stack on FB15k-237 with a trained DistMult | `fb15k237_clqa` |
| Run the temporal stack on ICEWS with a trained TComplEx | `icews14_temporal_clqa` |

## `taxonomy_query`: the connectives on a toy graph

```bash
cargo run --release --example taxonomy_query
```

```text
1p   dog is_a ?                      [Godel      ]  -> mammal (1.00)
2p   dog is_a ? is_a ?               [Product    ]  -> animal (1.00)
2i   (dog is_a ?) AND (cat is_a ?)   [Godel      ]  -> mammal (1.00)
2u   (dog is_a ?) OR (sparrow is_a ?) [Godel      ]  -> mammal (1.00), bird (1.00)
not  (dog eats ?) AND NOT (cat eats ?) [Lukasiewicz]  -> plant (0.30)
```

## `temporal_query`: interval-validity hops on a hand-built graph

Time windows registered as virtual relation ids; the two-terms query is an
ordinary intersection of two differently-windowed hops.

```bash
cargo run --release --example temporal_query
```

```text
held office in the 1990s                                -> alice (1.00)
held office before 1990 AND after 2000 (two terms)      -> bob (1.00)
held office after 2000 AND member of party_x            -> bob (0.90), carol (0.90)
```

## `fb15k237_clqa`: the CQD recipe end to end

Trained DistMult (tranz CLI) supplies atom degrees; queries compose in the
Product and Gödel algebras; hard answers only, filtered; `!` types are
non-reducible (every atom needs a held-out edge, per the ICLR 2025
reducibility critique). Closes with a witness and conformal coverage.

```bash
scripts/fetch_fb15k237.sh   # then train per the example header
cargo run --release --features tranz --example fb15k237_clqa
```

```text
type    n    MRR(P)    H@1    H@3   H@10     MRR(G)   H@10
1p    200     0.378  0.284  0.419  0.581      0.378  0.581
2p    165     0.236  0.141  0.260  0.448      0.220  0.407
2i    200     0.534  0.405  0.614  0.777      0.460  0.660
3i    200     0.688  0.557  0.768  0.914      0.548  0.774
2p!   168     0.262  0.172  0.282  0.447      0.236  0.421
2i!   200     0.383  0.270  0.420  0.630      0.409  0.613
3i!   200     0.415  0.286  0.471  0.707      0.468  0.724

conformal 1p answer sets: qhat 0.1004 on 300 valid pairs; held-out coverage 84% (nominal 80%)
```

The plain-to-`!` drop (2i 0.534 -> 2i! 0.383) is the honest measure of
multi-hop composition.

## `icews14_temporal_clqa`: the temporal stack on real events

Trained TComplEx (tranz 0.7.1, with the Lacroix et al. regularizers;
link-prediction test MRR 0.520) supplies time-scoped atom degrees; windowed
and not-during query types run against an exact `TemporalKg` oracle; closes
with a witness, a time-projection demo, and conformal coverage.

```bash
scripts/fetch_icews14.sh    # then train per the example header
cargo run --release --features tranz --example icews14_temporal_clqa
```

```text
type      n    MRR(P)    H@1    H@3   H@10     MRR(G)   H@10
1p       22     0.271  0.149  0.348  0.530      0.271  0.530
1p-t     86     0.440  0.337  0.500  0.605      0.440  0.605
2i-t     79     0.843  0.759  0.911  0.975      0.662  0.785
2i-t!    91     0.674  0.560  0.758  0.857      0.607  0.725
2u-t     49     0.187  0.095  0.196  0.394      0.210  0.426

time projection: when did (Ministry (Afghanistan), Make statement, Afghanistan) hold?
  predicted top days: 2014-03-10 2014-01-05 2014-08-19 (true: 2014-03-06)

conformal windowed-1p answer sets: qhat 0.3372 on 300 valid pairs; held-out coverage 81% (nominal 80%)
```

ICEWS05-15 (5x the quads, an 11x time axis) runs through the same harness:

```bash
scripts/fetch_icews0515.sh  # then train per the example header (batch 2048)
ICEWS_DATA=data/icews05-15 ICEWS_EMB=data/icews0515-tcomplex \
  cargo run --release --features tranz --example icews14_temporal_clqa
```
