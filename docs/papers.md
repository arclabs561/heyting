# Implementation Notes

This file records what each reference contributed to the implementation. It is
not required reading for using the crate.

### Metamathematics of Fuzzy Logic (Hajek, 1998)

Organizes fuzzy logics algebraically. Truth values live in `[0, 1]`; a
continuous t-norm `⊗` (associative, commutative, monotone, unit 1) plays
conjunction, and each t-norm determines a unique residuum `→` through the
adjunction `a ⊗ b ≤ c` iff `a ≤ b → c`. Modus ponens survives as the
residuation inequality `a ⊗ (a → b) ≤ b`. Three t-norms are fundamental
(Godel `min`, Product, Lukasiewicz), and every continuous t-norm is
assembled from them by ordinal sums, which is why these three are the named
logics everywhere in the field.

heyting takes: the `Truth` trait is a residuated lattice; the residuation
law is property-tested for every algebra; the three named algebras are
these three logics, and the residuum is the Heyting implication the crate
is named for.

### Provenance semirings (Green, Karvounarakis, Tannen, PODS 2007)

Annotate every source tuple of a database with an element of some structure
K and push the annotations through positive relational algebra. The main
theorem: the algebra's identities (the ones query rewriting relies on) hold
iff K is a commutative semiring. Provenance polynomials `N[X]` are the
universal such semiring: evaluate once with indeterminates and any concrete
semiring reading (cost, trust, probability, fuzzy degree) factors through
the polynomial by a homomorphism. "Why-provenance" (which source tuples
support an answer) is a quotient of that polynomial.

heyting takes: when an algebra is a semiring, this machinery applies to
query degrees. On `[0, 1]`, any t-norm paired with `max` is a semiring
(monotonicity gives distributivity), so `Godel` and `Viterbi` qualify;
Product's and Lukasiewicz's own disjunctions break distributivity
(counterexamples in the `truth` module docs), so witness extraction is
restricted to the semiring algebras.

### Semiring parsing (Goodman, 1999)

Shows the classic dynamic-programming parsing algorithms are one algorithm
parameterized by a semiring: run the same recursion over `(+, ×)` and you
get total probability; over `(max, ×)` and you get the probability of the
single best derivation. That `(max, ×)` structure is the Viterbi semiring.
The property doing the work is selectivity: `max` returns one of its
operands, so the computed value is always realized by one derivation.

heyting takes: the `Viterbi` algebra, and `SelectiveOr` as the exact
condition under which a witness's degree equals the engine's degree.

### Query2box (Ren, Hu, Leskovec, ICLR 2020)

Queries as boxes: anchors are points, each relation hop translates and
widens a box, conjunction is box intersection, and answers are the entities
inside the final box. The result that shapes everything downstream is
negative: a single fixed-dimension region cannot represent arbitrary
unions; the embedding dimension must grow with the number of entities. So
union queries are rewritten to disjunctive normal form and the union is
taken last, over answer sets.

heyting takes: `BoxModel` scores in this geometry, and `BoxDnf`
materializes queries as exact box intersections with DNF unions because the
theorem forces DNF; negation is refused because boxes are not closed under
complement.

### Beta embeddings (Ren, Leskovec, NeurIPS 2020)

BetaE embeds entities and queries as Beta distributions, which are closed
under a complement operation, making it the first embedding method to
support full first-order negation. Its equally durable contribution is the
evaluation protocol used by the field since: split each query's answers
into easy (reachable in the training graph) and hard (requiring held-out
edges), and report filtered ranking metrics over the hard answers only.

heyting takes: the `eval` module implements that protocol. The
distributional geometry itself is deliberately not adopted.

### Complex query answering with neural link predictors (Arakelyan, Daza, Minervini, Cochez, ICLR 2021)

CQD: skip complex-query training entirely. Take a link predictor trained
only on one-hop facts, squash its scores into `[0, 1]`, and answer complex
queries at inference time by composing atom scores through t-norms and
t-conorms, searching intermediate entities by beam or continuous
optimization. It matched or beat query-trained systems with orders of
magnitude less training signal.

heyting takes: its whole execution model. The engine is this idea
generalized to any scorer through `AtomicScorer` and a typed algebra
instead of ad-hoc fuzzy operations. It is also why this crate has no
composition-aware trainer.

### Rethinking complex queries on knowledge graphs with neural link predictors (Yin, Wang, Song, ICLR 2024)

Separates query classes the field had been conflating: "tree-form" queries,
whose query graph is a tree (all the classic benchmarks), versus full
existential first-order queries with one free variable, which include
cycles and multi-edge joins. Tree-form admits exact bottom-up evaluation;
cyclic queries require search or inference over the query graph, and the
paper gives a fuzzy-inference solver for the full class.

heyting takes: honest scope. `Query` is a tree ADT, which is exactly the
tree-form fragment; cyclic evaluation is declared out of scope rather than
approximated silently.

### Algorithmic Learning in a Random World (Vovk, Gammerman, Shafer, 2005)

The conformal prediction book. If calibration and test data are
exchangeable, ranking a test point's nonconformity score among the
calibration scores yields a valid p-value, so prediction sets built from
those ranks contain the truth with any chosen probability. The guarantee
needs no assumption about the model or the data distribution beyond
exchangeability, and it holds at finite sample sizes.

heyting takes: the coverage guarantee behind the `conformal` module.

### A gentle introduction to conformal prediction (Angelopoulos, Bates, 2021)

The practical recipe the field standardized on. Split conformal: hold out a
calibration set, compute a nonconformity score per calibration pair, take
the `⌈(n+1)(1−α)⌉/n` empirical quantile, and the prediction set for a new
input is everything scoring within that quantile. Coverage `1 − α` follows
from exchangeability alone.

heyting takes: this exact procedure over `(query, answer)` scores; the
README quotes the measured FB15k-237 coverage.

### Advancing abductive reasoning in knowledge graphs through complex logical hypothesis generation (Bai et al., ACL 2024)

Defines the abduction task over knowledge graphs: given a set of observed
entities, produce the logical hypothesis (a complex query) whose answer set
best explains them, with quality measured by answer-set overlap against the
observation. Their solver is a trained generative model refined with
feedback from the graph.

heyting takes: the task and the scoring shape. `abduce` is the symbolic
version: sweep template hypotheses (one-hop atoms, pairwise conjunctions),
score by fuzzy Jaccard, return the best. No learned generator.

### Tensor decompositions for temporal knowledge base completion (Lacroix, Obozinski, Usunier, ICLR 2020)

Expresses a temporal knowledge base as an order-4 tensor and scores quads
with TComplEx: `Re(⟨h, r, conj(t), w_τ⟩)`, where the timestamp embedding
modulates the relation (equivalently either entity) componentwise.
TNTComplEx splits the relation into temporal and static parts for
heterogeneous KBs where many predicates never change; on all-event data
(ICEWS) it matches plain TComplEx, so the split earns its parameters only
when non-temporal facts dominate. The paper's real lever is
regularization: the weighted nuclear-3 variational form Ω³ penalizes the
`(relation ∘ timestamp)` product as one factor — unfolding the tensor
along the predicate and time modes shows this weights by the joint
predicate-time marginal rather than the product of marginals — and the
smoothness prior Λ_p penalizes the discrete derivative of the timestamp
table. Together they are worth about five MRR points on ICEWS05-15
(their Table 3). Training is tail-only cross-entropy with reciprocal
relations; a second "time-tube" cross-entropy (their Eq. 7) teaches the
timestamp-answering direction at about one MRR point's cost to entity
ranking.

heyting takes: the trained scorer behind `adapters::TemporalPointModel`.
tranz implements the model, both regularizers, and a `score_all_times`
seam (the Eq. 7 direction) as `tranz::temporal`.

### TFLEX: temporal feature-logic embedding framework (Lin et al., NeurIPS 2023)

Defines multi-hop logical reasoning over temporal KGs. Queries answer
entity sets or timestamp sets; the computation-graph semantics makes
entity projection existential over a timestamp set, gives set complement
on either sort, and adds three set-level temporal operators:
`After(S) = {t' : t' > max(S)}`, `Before(S) = {t' : t' < min(S)}`, and
`Between(S₁, S₂)` as their intersection. Their solver embeds each query
as (entity feature, entity logic, time feature, time logic) with fuzzy
vector logic on the logic parts and translation-style MLP projections;
ablations show removing either the time part or the logic part costs
real accuracy, and grafting a static geometric query embedding onto the
time part conflicts semantically.

heyting takes: the crisp set semantics, implemented symbolically —
`TimeSet::after_all` / `before_all` / `between_all` are the three
operators exactly, entity hops are existential over the set, and
complement is native to the bitset carrier. Not taken: the neural query
embedding itself, and timestamp-answering queries (their time
projection), which remain the named gap.
