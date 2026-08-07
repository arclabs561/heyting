# heyting examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs. The two dataset examples are data-gated: without
their data they print fetch/train instructions and exit 0.

## Which example should I run?

| I want to... | Example |
|---|---|
| See the connectives on a ten-entity toy taxonomy | `taxonomy_query` |
| See interval-validity temporal hops, no training | `temporal_query` |
| Calibrate raw one-hop scores before query evaluation | `raw_calibration` |
| Run the full stack on FB15k-237 with a trained DistMult | `fb15k237_clqa` |
| Run the temporal stack on ICEWS with a trained TComplEx | `icews14_temporal_clqa` |
| Score the published BetaE benchmark files, all 14 types | `betae_fb15k237` |
| Graded EL++ subsumption complex queries over region boxes | `el_clqa` |
| Conformal answer sets over a faithful EL++ model | `el_clqa_conformal` |
| Conformal LCA sets over the off-seam gated readout | `el_clqa_gated_conformal` |

## `el_clqa`: graded EL++ subsumption CLQA over region embeddings

Entities are concept boxes (as from a Box2EL-style trainer); the atomic
relation is graded subsumption `C ⊑ ?`. Composes subsumption degrees through
the residuated logic. Two queries: common superclasses of `Dog` and `Cat`,
and dog-only superclasses (subsumption with negation).

```bash
cargo run --release --features subsume --example el_clqa
```

```text
Query A: X such that Dog ⊑ X AND Cat ⊑ X  (common superclasses)
  Godel:       Animal=1.000  Mammal=0.607  Dog=0.018
  Product:     Animal=1.000  Mammal=0.368  Dog=0.018
  Lukasiewicz: Animal=1.000  Mammal=0.213  Dog=0.018
  -> Mammal graded truth by logic: Godel 0.607 > Product 0.368 > Lukasiewicz 0.213
All assertions passed: graded EL++ subsumption CLQA over region embeddings.
```

## `el_clqa_conformal`: conformal sets over a faithful EL++ model

Graded subsumption ranks superclasses but not how many to trust; split
conformal prediction turns them into coverage-guaranteed answer sets, using
the always-on `FaithfulBoxModel` adapter (no ecosystem crate). 13 concepts,
20 subsumption pairs.

```bash
cargo run --release --example el_clqa_conformal
```

```text
13 concepts, 20 subsumption pairs (10 calibration, 10 test)
Sample true-subsumption degrees for leaf 5: ⊑mid=1.000  ⊑Thing=0.819

alpha     1-alpha      q_hat    empirical
0.30         0.70      0.632        1.000
0.20         0.80      0.632        1.000
0.10         0.90      0.632        1.000

Conformal superclass set for leaf 5 (alpha=0.1): 1=1.00 5=1.00 0=0.82
(true ancestors: mid 1 and Thing 0)
All assertions passed: conformal answer sets over a faithful EL++ model.
```

## `el_clqa_gated_conformal`: conformal LCA sets over a gated readout

The conjunctive least-common-ancestor readout is off the atomic seam, so this
exercises the scorer-agnostic conformal core on `subsume::clqa::BoxClqa`
directly.

```bash
cargo run --release --features subsume --example el_clqa_gated_conformal
```

```text
15 concepts, 28 conjunctive leaf-pair queries (14 calibration, 14 test)
gated readout top-1 LCA accuracy: 28/28

alpha     1-alpha      q_hat    empirical    mean|set|
0.30         0.70      0.976        1.000         1.14
0.20         0.80      0.976        1.000         1.14
0.10         0.90      0.976        1.000         1.14

Conformal LCA set for (leaf 7, leaf 9) at alpha=0.1: 1=0.29
(true LCA: super 1 )
All assertions passed: conformal LCA sets over the off-seam gated readout.
```

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

## `raw_calibration`: raw scores into query degrees

```bash
cargo run --release --example raw_calibration
```

```text
(dog is_a ?) AND (cat is_a ?)
  #1 mammal   0.881
  #2 animal   0.119
  #3 dog      0.000
```

## `fb15k237_clqa`: the CQD recipe end to end

Trained DistMult (tranz CLI) supplies atom degrees; queries compose in the
Product and Gödel algebras; hard answers only, filtered; `!` types are
non-reducible (every atom needs a held-out edge, per the ICML 2025
reducibility critique). Closes with a witness and conformal coverage.

```bash
scripts/fetch_fb15k237.sh   # then train per the example header
cargo run --release --features tranz --example fb15k237_clqa
```

```text
type    n    MRR(P)    H@1    H@3   H@10     MRR(G)   H@10
1p    200     0.389  0.294  0.441  0.555      0.389  0.555
2p    165     0.238  0.157  0.269  0.393      0.224  0.375
2i    200     0.566  0.429  0.659  0.822      0.467  0.684
3i    200     0.731  0.600  0.843  0.948      0.553  0.767
2p!   168     0.232  0.155  0.257  0.374      0.216  0.354
2i!   200     0.361  0.257  0.383  0.606      0.388  0.604
3i!   200     0.385  0.277  0.416  0.621      0.426  0.661

conformal 1p answer sets: qhat 0.3043 on 300 valid pairs; held-out coverage 80% (nominal 80%)
```

The model is a DistMult d 256 trained with reciprocals and the weighted
nuclear-3 regularizer (`--reciprocals --n3-reg 0.0025 --init-scale 0.01`).
The plain-to-`!` drop (2i 0.566 -> 2i! 0.361) is the honest measure of
multi-hop composition.

## `icews14_temporal_clqa`: the temporal stack on real events

Trained TComplEx (tranz >= 0.7.2, with the Lacroix et al. regularizers;
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

## `betae_fb15k237`: the published benchmark files, evaluated exactly

Loads the KGReasoning pickles (queries + easy/hard answer sets) and scores
all 14 BetaE query types file-exactly, so rows compare directly against the
BetaE / CQD / QTO FB15k-237 tables (theirs use ComplEx-N3 at d >= 1000;
compare shapes, not absolutes). Negation rows exclude the negated branch's
top-50 candidates via a crisp `Query::given` mask; soft `1 - sigmoid`
negation over uncalibrated degrees measures at the random floor.

```bash
scripts/fetch_betae_fb15k237.sh   # ~1.4 GB download
cargo run --release --features tranz --example betae_fb15k237
```

```text
type     n        MRR    H@1    H@3   H@10     MRR(G)   H@10
1p     300   0.329 (P)  0.235  0.374  0.500      0.329  0.500
2p     300   0.025 (P)  0.010  0.024  0.053      0.022  0.044
3p     300   0.040 (P)  0.018  0.036  0.076      0.036  0.073
2i     300   0.167 (P)  0.097  0.194  0.289      0.153  0.270
3i     300   0.180 (P)  0.122  0.199  0.282      0.157  0.259
pi     300   0.120 (P)  0.071  0.121  0.213      0.102  0.178
ip     300   0.126 (P)  0.075  0.129  0.230      0.088  0.193
2u     300   0.071 (P)  0.025  0.073  0.148      0.070  0.145
up     300   0.034 (P)  0.018  0.031  0.059      0.029  0.054
2in    300   0.010(P¬)  0.000  0.002  0.007      0.000  0.000
3in    300   0.011(P¬)  0.004  0.005  0.011      0.000  0.000
inp    300   0.016(P¬)  0.005  0.008  0.035      0.000  0.000
pin    300   0.009(P¬)  0.002  0.003  0.018      0.000  0.000
pni    300   0.022(P¬)  0.006  0.009  0.035      0.000  0.000
```
