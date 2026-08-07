//! The query algebra: a compositional [`Query`] evaluated over an
//! [`AtomicScorer`] in a chosen [`Truth`] algebra.
//!
//! This mirrors `tranz::query` (CQD-Beam, Arakelyan et al. 2021) but abstracts
//! the model behind [`AtomicScorer`], so the same engine answers queries over
//! point embeddings (tranz), region embeddings (subsume), or any other source
//! of one-hop membership degrees. The geometry lives entirely behind
//! `project`; the engine only ever sees degrees in `[0, 1]`.

use crate::truth::Truth;
use std::collections::HashMap;

/// Ordering convention for raw model scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawScoreOrder {
    /// Larger raw scores mean stronger membership.
    HigherIsBetter,
    /// Smaller raw scores mean stronger membership.
    LowerIsBetter,
}

/// Uncalibrated one-hop scores for `(anchor, relation, ?)`.
///
/// Query evaluation consumes calibrated degrees from
/// [`AtomicScorer::project`]. This raw form is for calibrators and diagnostics
/// that need the model's native score scale before it is mapped to `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RawProjection {
    /// Raw score per entity.
    pub scores: Vec<f32>,
    /// Whether larger or smaller scores are better in `scores`.
    pub order: RawScoreOrder,
}

impl RawProjection {
    /// Create a raw projection with an explicit score order.
    pub fn new(scores: Vec<f32>, order: RawScoreOrder) -> Self {
        Self { scores, order }
    }

    /// Number of entity scores.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether no scores are present.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Return a score where larger means stronger membership.
    pub fn oriented_score(&self, score: f32) -> f32 {
        match self.order {
            RawScoreOrder::HigherIsBetter => score,
            RawScoreOrder::LowerIsBetter => -score,
        }
    }

    /// Return all scores oriented so larger means stronger membership.
    pub fn oriented_scores(&self) -> Vec<f32> {
        self.scores
            .iter()
            .map(|&score| self.oriented_score(score))
            .collect()
    }
}

/// A source of atomic one-hop answers: the only thing the engine needs from a
/// model.
///
/// `project(anchor, relation)` returns, for every entity `t`, the degree in
/// `[0, 1]` to which `(anchor, relation, t)` holds — i.e. the membership of `t`
/// in the answer set of the atomic query `(anchor, relation, ?)`. A point model
/// produces this by normalizing link-prediction scores; a region model by
/// scoring containment of each entity in the projected region.
pub trait AtomicScorer {
    /// Number of entities; the length of every [`AtomicScorer::project`] result.
    fn num_entities(&self) -> usize;

    /// Membership degrees in `[0, 1]` for all entities as the answer to
    /// `(anchor, relation, ?)`. Higher means more strongly an answer.
    fn project(&self, anchor: usize, relation: usize) -> Vec<f32>;

    /// Membership degrees for many same-relation anchors.
    ///
    /// The outer vec follows `anchors`; each inner vec is a dense projection
    /// for the corresponding anchor. Scorers with native batched projections
    /// should override this method.
    fn project_batch(&self, anchors: &[usize], relation: usize) -> Vec<Vec<f32>> {
        anchors
            .iter()
            .map(|&anchor| self.project(anchor, relation))
            .collect()
    }

    /// Native uncalibrated scores for `(anchor, relation, ?)`, when available.
    ///
    /// The default is `None` because some scorers are already degree-valued or
    /// have no meaningful raw-score surface. Calibrators use this hook and fall
    /// back to [`AtomicScorer::project`] when it is unavailable.
    fn project_raw(&self, _anchor: usize, _relation: usize) -> Option<RawProjection> {
        None
    }

    /// Native uncalibrated scores for many same-relation anchors.
    ///
    /// The default calls [`AtomicScorer::project_raw`] for each anchor and
    /// returns `None` if any anchor lacks raw scores.
    fn project_raw_batch(&self, anchors: &[usize], relation: usize) -> Option<Vec<RawProjection>> {
        let mut batch = Vec::with_capacity(anchors.len());
        for &anchor in anchors {
            batch.push(self.project_raw(anchor, relation)?);
        }
        Some(batch)
    }

    /// Membership degrees for `candidates` only, aligned with `candidates`.
    ///
    /// The default gathers from the dense [`project`](AtomicScorer::project);
    /// embedding-backed scorers should override it to score only the subset,
    /// which is where the [`crate::prune`] path gets its speedup.
    fn project_subset(&self, anchor: usize, relation: usize, candidates: &[usize]) -> Vec<f32> {
        let dense = self.project(anchor, relation);
        candidates
            .iter()
            // Out-of-range candidate = degree 0.0 by the engine's padding
            // convention (same as `answer_query`), not an error sentinel.
            .map(|&i| dense.get(i).copied().unwrap_or(0.0))
            .collect()
    }
}

/// Evaluation knobs.
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Beam width for the existential over each intermediate variable in a
    /// chain: only the top-`beam_k` intermediates are expanded. Higher improves
    /// recall at `O(beam_k · |E|)` per hop. Default: 128.
    pub beam_k: usize,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self { beam_k: 128 }
    }
}

impl QueryConfig {
    /// Expand every positive intermediate in chain projections.
    ///
    /// This disables beam truncation by setting [`QueryConfig::beam_k`] to
    /// `usize::MAX`. It is useful for exact small-graph checks and for the
    /// sparse Viterbi/QTO path when the candidate supports are already bounded.
    pub const fn exact() -> Self {
        Self { beam_k: usize::MAX }
    }
}

/// A complex logical query as a computation tree over atomic projections.
///
/// # Scope: the tree-form fragment
///
/// This type is a tree, so it expresses exactly the *tree-form* fragment of
/// existential first-order queries — the acyclic class where evaluation is
/// tractable (Yannakakis) and the pruned evaluator is exact. Cyclic or
/// multi-anchor-join EFO1 queries (two atoms constraining the same pair of
/// variables) are not constructible here; supporting them is a search
/// problem (QTO/FIT-style), deliberately out of scope.
///
/// Build with the constructors ([`Query::anchor`], [`Query::then`], etc.) and
/// evaluate with [`answer_query`] / [`answer_query_topk`]. The connectives
/// realize existential positive first-order logic plus negation and the Heyting
/// implication, each interpreted through the chosen [`Truth`] algebra.
///
/// # Two disjunctions
///
/// Disjunction appears twice with deliberately different aggregation. An
/// explicit [`Query::Union`] folds its branches with the algebra's t-conorm
/// `⊕` ([`Truth::or`]), so under [`crate::Product`] or [`crate::Lukasiewicz`]
/// it is *not* a plain `max`. The existential over a chain's intermediate
/// variable ([`Query::Project`]) instead always aggregates with `max` (the
/// Gödel join), in every algebra, following CQD-Beam. The two therefore differ
/// outside [`crate::Godel`]: explicit `Union` follows the algebra, the chain
/// existential follows CQD-Beam.
///
/// # Warning
///
/// The constructors ([`Query::intersection`], [`Query::union`]) enforce that a
/// branch list is non-empty, but building the [`Query::Intersection`] /
/// [`Query::Union`] variants directly bypasses that check. An empty
/// `Intersection` evaluates to `⊤` (the conjunction's unit) and an empty
/// `Union` to `⊥` (the disjunction's unit) at every entity; prefer the
/// constructors.
#[derive(Debug, Clone)]
pub enum Query {
    /// Atomic `(entity, relation, ?)`.
    Anchor {
        /// Anchor (head) entity id.
        entity: usize,
        /// Relation id.
        relation: usize,
    },
    /// Chain: evaluate `inner`, then existentially project the intermediate
    /// answers through `relation` (`∃V. inner(V) ∧ (V, relation, ?)`).
    Project {
        /// Sub-query producing the intermediate variable's degrees.
        inner: Box<Query>,
        /// Relation to project through.
        relation: usize,
    },
    /// Conjunction `⋀`: combine branches with the t-norm `⊗`.
    Intersection {
        /// Branches to intersect (one or more).
        branches: Vec<Query>,
    },
    /// Disjunction `⋁`: combine branches with the t-conorm `⊕`.
    Union {
        /// Branches to union (one or more).
        branches: Vec<Query>,
    },
    /// Negation `¬`: the algebra's pseudo-complement.
    Negation {
        /// Sub-query to negate.
        inner: Box<Query>,
    },
    /// Implication `premise → conclusion`: the Heyting residuum, per entity.
    ///
    /// Answers "entities for which, to the degree they satisfy `premise`, they
    /// also satisfy `conclusion`" — a conditional/filtering query. This is the
    /// namesake operation; under [`crate::Godel`] it is the genuine
    /// intuitionistic implication.
    Implication {
        /// Antecedent sub-query.
        premise: Box<Query>,
        /// Consequent sub-query.
        conclusion: Box<Query>,
    },
    /// A precomputed membership leaf: degree `degrees[e]` for entity `e`
    /// (missing entries are `0`).
    ///
    /// This is how facts that are not relation hops enter a query — most
    /// usefully numeric-literal constraints ("born after 1970"), where the
    /// degree vector encodes an interval or half-space over an attribute
    /// (crisp `0`/`1`, or a soft ramp near the boundary). Intersect it with
    /// relation hops to get literal-constrained queries; see
    /// [`Query::given`].
    Given {
        /// Per-entity membership degrees in `[0, 1]`.
        degrees: Vec<f32>,
    },
}

impl Query {
    /// Atomic one-hop query `(entity, relation, ?)`.
    pub fn anchor(entity: usize, relation: usize) -> Self {
        Query::Anchor { entity, relation }
    }

    /// Chain a relation onto this query: `self → relation → ?`.
    pub fn then(self, relation: usize) -> Self {
        Query::Project {
            inner: Box::new(self),
            relation,
        }
    }

    /// Conjunction of branches.
    ///
    /// # Panics
    /// Panics if `branches` is empty.
    pub fn intersection(branches: Vec<Query>) -> Self {
        assert!(!branches.is_empty(), "intersection requires a branch");
        Query::Intersection { branches }
    }

    /// Disjunction of branches.
    ///
    /// # Panics
    /// Panics if `branches` is empty.
    pub fn union(branches: Vec<Query>) -> Self {
        assert!(!branches.is_empty(), "union requires a branch");
        Query::Union { branches }
    }

    /// Negate this query (the algebra's pseudo-complement).
    pub fn negate(self) -> Self {
        Query::Negation {
            inner: Box::new(self),
        }
    }

    /// Build `self → conclusion` (the Heyting residuum).
    pub fn implies(self, conclusion: Query) -> Self {
        Query::Implication {
            premise: Box::new(self),
            conclusion: Box::new(conclusion),
        }
    }

    /// A precomputed membership leaf; degrees are clamped to `[0, 1]`.
    ///
    /// The literal-constraint pattern: encode "attribute in `[lo, hi]`" as a
    /// degree vector and conjoin it with relation hops.
    ///
    /// ```
    /// use heyting::Query;
    ///
    /// // "born after 1970" over a birth-year attribute (None = unknown = 0).
    /// let birth_years = [Some(1962.0_f32), Some(1975.0), None];
    /// let constraint = Query::given(
    ///     birth_years
    ///         .iter()
    ///         .map(|y| match y {
    ///             Some(y) if *y > 1970.0 => 1.0,
    ///             _ => 0.0,
    ///         })
    ///         .collect(),
    /// );
    /// let q = Query::intersection(vec![Query::anchor(0, 0), constraint]);
    /// # let _ = q;
    /// ```
    pub fn given(mut degrees: Vec<f32>) -> Self {
        for d in &mut degrees {
            *d = if d.is_finite() {
                d.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        Query::Given { degrees }
    }
}

/// Answer `query` in the truth algebra `T`, returning a degree in `[0, 1]` for
/// every entity (length `scorer.num_entities()`).
///
/// `T` selects the logic: [`crate::Godel`] for intersections, [`crate::Product`]
/// for chains, [`crate::Lukasiewicz`] for negation-bearing queries (see the
/// [`crate::truth`] module).
pub fn answer_query<T: Truth>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
) -> Vec<f32> {
    let n = scorer.num_entities();
    eval_cached::<T>(scorer, query, config, n, &mut AtomicCache::new())
}

/// Answer many queries in the truth algebra `T`, returning a degree vector
/// (length `scorer.num_entities()`) per query, in input order.
///
/// Within a batch, atomic `(anchor, relation)` projections are computed once
/// and reused across every query that needs them. For a batch that shares
/// anchors or relations (the common eval-loop shape), this is much cheaper
/// than calling [`answer_query`] per query; a batch of one is equivalent to
/// [`answer_query`]. Each query is still evaluated independently and produces
/// exactly the degrees [`answer_query`] would.
pub fn answer_queries<T: Truth>(
    scorer: &dyn AtomicScorer,
    queries: &[Query],
    config: &QueryConfig,
) -> Vec<Vec<f32>> {
    let n = scorer.num_entities();
    let mut cache = AtomicCache::new();
    queries
        .iter()
        .map(|q| eval_cached::<T>(scorer, q, config, n, &mut cache))
        .collect()
}

/// Memo of dense `(anchor, relation)` projections, shared across a batch of
/// queries. Only atomic leaves are cached; `project_batch` already batches
/// the per-hop projections, and caching the deterministic dense leaves is
/// what collapses repeated work across queries sharing an anchor+relation.
struct AtomicCache {
    projections: HashMap<(usize, usize), Vec<f32>>,
}

impl AtomicCache {
    fn new() -> Self {
        Self {
            projections: HashMap::new(),
        }
    }

    fn project(
        &mut self,
        scorer: &dyn AtomicScorer,
        anchor: usize,
        relation: usize,
        n: usize,
    ) -> Vec<f32> {
        use std::collections::hash_map::Entry;
        match self.projections.entry((anchor, relation)) {
            // Cache hit: the costly `scorer.project` is skipped; only the
            // cheap dense memcpy return matters. This is the whole point of
            // the batch API -- a batch sharing (anchor, relation) computes
            // that projection once.
            Entry::Occupied(hit) => hit.get().clone(),
            Entry::Vacant(slot) => {
                let mut s = scorer.project(anchor, relation);
                s.resize(n, 0.0);
                slot.insert(s).clone()
            }
        }
    }
}

/// Like [`eval`], but memoizes atomic leaves through `cache`, which is shared
/// across the queries of one batch.
fn eval_cached<T: Truth>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
    n: usize,
    cache: &mut AtomicCache,
) -> Vec<f32> {
    match query {
        Query::Anchor { entity, relation } => cache.project(scorer, *entity, *relation, n),
        Query::Project { inner, relation } => {
            let inner_scores = eval_cached::<T>(scorer, inner, config, n, cache);
            project::<T>(scorer, &inner_scores, *relation, config, n)
        }
        Query::Intersection { branches } => {
            let mut acc = vec![T::top(); n];
            for branch in branches {
                let s = eval_cached::<T>(scorer, branch, config, n, cache);
                for (a, b) in acc.iter_mut().zip(s.iter()) {
                    *a = T::and(*a, *b);
                }
            }
            acc
        }
        Query::Union { branches } => {
            let mut acc = vec![T::bot(); n];
            for branch in branches {
                let s = eval_cached::<T>(scorer, branch, config, n, cache);
                for (a, b) in acc.iter_mut().zip(s.iter()) {
                    *a = T::or(*a, *b);
                }
            }
            acc
        }
        Query::Negation { inner } => {
            let mut s = eval_cached::<T>(scorer, inner, config, n, cache);
            for x in &mut s {
                *x = T::neg(*x);
            }
            s
        }
        Query::Implication {
            premise,
            conclusion,
        } => {
            let p = eval_cached::<T>(scorer, premise, config, n, cache);
            let c = eval_cached::<T>(scorer, conclusion, config, n, cache);
            p.iter()
                .zip(c.iter())
                .map(|(&pi, &ci)| T::residuum(pi, ci))
                .collect()
        }
        Query::Given { degrees } => {
            let mut s = degrees.clone();
            s.resize(n, 0.0);
            s
        }
    }
}

/// Answer `query` and return the top-`k` `(entity, degree)` pairs, best first.
pub fn answer_query_topk<T: Truth>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
    k: usize,
) -> Vec<(usize, f32)> {
    top_k_descending(&answer_query::<T>(scorer, query, config), k)
}

/// Existential projection over a chain hop.
///
/// `∃V. inner(V) ∧ (V, relation, ?)`. Expands the top-`beam_k` intermediates,
/// conjoins each one's degree (via the t-norm) with its projected tail degrees,
/// and takes the supremum (`max`) over intermediates — the existential
/// quantifier is the lattice join, which is `max` on `[0, 1]` for every
/// [`Truth`] algebra.
fn project<T: Truth>(
    scorer: &dyn AtomicScorer,
    inner_scores: &[f32],
    relation: usize,
    config: &QueryConfig,
    n: usize,
) -> Vec<f32> {
    let beam: Vec<_> = top_k_descending(inner_scores, config.beam_k)
        .into_iter()
        .filter(|&(_, v_score)| v_score > 0.0)
        .collect();
    let anchors: Vec<_> = beam.iter().map(|(v, _)| *v).collect();
    let projections = scorer.project_batch(&anchors, relation);
    let mut out = vec![0.0_f32; n];
    for ((_, v_score), tails) in beam.iter().zip(projections.iter()) {
        for (t, &tail) in tails.iter().enumerate().take(n) {
            let combined = T::and(*v_score, tail);
            if combined > out[t] {
                out[t] = combined;
            }
        }
    }
    out
}

pub(crate) fn top_k_descending(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }

    let mut idx: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k, degree_id_desc_order);
        idx.truncate(k);
    }
    idx.sort_unstable_by(degree_id_desc_order);
    idx
}

fn degree_id_desc_order(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::truth::{Godel, Lukasiewicz, Product};
    use crate::FuzzyKg;

    // A tiny taxonomy: 0=animal, 1=mammal, 2=bird, 3=dog, 4=cat, 5=sparrow.
    // relation 0 = is_a (child -> parent). relation 1 = eats.
    fn taxonomy() -> FuzzyKg {
        let mut kg = FuzzyKg::new(6);
        // is_a edges (relation 0): dog/cat -> mammal, sparrow -> bird, mammal/bird -> animal.
        kg.add_edge(3, 0, 1, 1.0); // dog is_a mammal
        kg.add_edge(4, 0, 1, 1.0); // cat is_a mammal
        kg.add_edge(5, 0, 2, 1.0); // sparrow is_a bird
        kg.add_edge(1, 0, 0, 1.0); // mammal is_a animal
        kg.add_edge(2, 0, 0, 1.0); // bird is_a animal
                                   // eats edges (relation 1): dog eats meat... reuse ids loosely for the test.
        kg.add_edge(3, 1, 4, 0.8); // dog eats cat (degree 0.8)
        kg
    }

    #[test]
    fn anchor_returns_direct_neighbours() {
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        // dog is_a ? -> mammal (entity 1).
        let q = Query::anchor(3, 0);
        let scores = answer_query::<Godel>(&kg, &q, &cfg);
        assert_eq!(scores.len(), 6);
        assert!((scores[1] - 1.0).abs() < 1e-6, "dog is_a mammal");
        assert!(scores[0].abs() < 1e-6, "not directly animal");
    }

    #[test]
    fn exact_config_disables_beam_truncation() {
        assert_eq!(QueryConfig::exact().beam_k, usize::MAX);
    }

    #[test]
    fn chain_2p_reaches_grandparent() {
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        // dog is_a ? is_a ? -> animal (entity 0), via mammal.
        let q = Query::anchor(3, 0).then(0);
        let scores = answer_query::<Product>(&kg, &q, &cfg);
        let top = top_k_descending(&scores, 1);
        assert_eq!(top[0].0, 0, "two-hop is_a from dog reaches animal");
    }

    #[test]
    fn intersection_keeps_only_shared_answers() {
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        // (dog is_a ?) AND (cat is_a ?) -> both are mammal.
        let q = Query::intersection(vec![Query::anchor(3, 0), Query::anchor(4, 0)]);
        let scores = answer_query::<Godel>(&kg, &q, &cfg);
        let top = top_k_descending(&scores, 1);
        assert_eq!(top[0].0, 1, "dog and cat agree on mammal");
    }

    #[test]
    fn union_includes_either_branch() {
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        // (dog is_a ?) OR (sparrow is_a ?) -> {mammal, bird}.
        let q = Query::union(vec![Query::anchor(3, 0), Query::anchor(5, 0)]);
        let scores = answer_query::<Godel>(&kg, &q, &cfg);
        assert!(scores[1] > 0.5, "mammal in union");
        assert!(scores[2] > 0.5, "bird in union");
    }

    #[test]
    fn negation_under_lukasiewicz_is_one_minus() {
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        let q = Query::anchor(3, 0); // dog is_a mammal (deg 1.0 at entity 1)
        let pos = answer_query::<Lukasiewicz>(&kg, &q, &cfg);
        let neg = answer_query::<Lukasiewicz>(&kg, &q.clone().negate(), &cfg);
        for i in 0..6 {
            assert!((pos[i] + neg[i] - 1.0).abs() < 1e-5, "involutive at {i}");
        }
    }

    #[test]
    fn implication_is_top_where_premise_below_conclusion() {
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        // (dog is_a ?) -> (cat is_a ?). For mammal both are 1.0, so residuum = 1.
        let q = Query::anchor(3, 0).implies(Query::anchor(4, 0));
        let scores = answer_query::<Godel>(&kg, &q, &cfg);
        assert!((scores[1] - 1.0).abs() < 1e-6, "1.0 → 1.0 = ⊤ at mammal");
    }

    /// Evaluate `premise → conclusion` at entity 1 where the premise holds to
    /// degree `p` and the conclusion to degree `c` there. Premise is relation 0
    /// and conclusion relation 1, both anchored at entity 0, so the two degrees
    /// are set independently by their edge weights.
    fn implication_at<T: Truth>(p: f32, c: f32) -> f32 {
        let mut kg = FuzzyKg::new(2);
        if p > 0.0 {
            kg.add_edge(0, 0, 1, p);
        }
        if c > 0.0 {
            kg.add_edge(0, 1, 1, c);
        }
        let q = Query::anchor(0, 0).implies(Query::anchor(0, 1));
        answer_query::<T>(&kg, &q, &QueryConfig::default())[1]
    }

    #[test]
    fn implication_is_vacuously_true_when_premise_is_zero() {
        // Premise 0 ≤ any conclusion, so a → b = ⊤ in every algebra.
        assert!((implication_at::<Godel>(0.0, 0.7) - 1.0).abs() < 1e-6);
        assert!((implication_at::<Product>(0.0, 0.7) - 1.0).abs() < 1e-6);
        assert!((implication_at::<Lukasiewicz>(0.0, 0.7) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn implication_is_top_when_premise_strictly_below_conclusion() {
        // p < c ⟹ residuum = ⊤ (reflexivity of the order) in every algebra.
        assert!((implication_at::<Godel>(0.3, 0.8) - 1.0).abs() < 1e-6);
        assert!((implication_at::<Product>(0.3, 0.8) - 1.0).abs() < 1e-6);
        assert!((implication_at::<Lukasiewicz>(0.3, 0.8) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn implication_pins_each_algebra_when_premise_exceeds_conclusion() {
        // p > c is where the three algebras genuinely differ; assert exact
        // values so a constant-⊤ implementation would fail all three.
        let (p, c) = (0.8_f32, 0.3_f32);
        assert!(
            (implication_at::<Godel>(p, c) - c).abs() < 1e-6,
            "Godel: a → b = b when a > b"
        );
        assert!(
            (implication_at::<Product>(p, c) - (c / p)).abs() < 1e-6,
            "Product: a → b = b / a when a > b"
        );
        assert!(
            (implication_at::<Lukasiewicz>(p, c) - (1.0 - p + c)).abs() < 1e-6,
            "Lukasiewicz: a → b = 1 − a + b"
        );
    }

    #[test]
    fn answer_queries_matches_per_query_answer_query() {
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        let queries = vec![
            Query::anchor(3, 0),
            Query::anchor(3, 0).then(0),
            Query::intersection(vec![Query::anchor(3, 0), Query::anchor(4, 0)]),
            Query::union(vec![Query::anchor(3, 0), Query::anchor(5, 0)]),
            Query::anchor(3, 0).negate(),
            Query::anchor(3, 0).implies(Query::anchor(4, 0)),
        ];
        let batched = answer_queries::<Godel>(&kg, &queries, &cfg);
        assert_eq!(batched.len(), queries.len());
        for (batch, q) in batched.iter().zip(&queries) {
            let single = answer_query::<Godel>(&kg, q, &cfg);
            assert_eq!(batch.len(), single.len());
            for (b, s) in batch.iter().zip(single.iter()) {
                assert!((b - s).abs() < 1e-6, "batch {b} != single {s}\nq={q:?}");
            }
        }
    }

    /// The whole point of the batch API: a shared atomic-leaf cache computes
    /// a reused (anchor, relation) projection once instead of once per query.
    #[test]
    fn answer_queries_dedups_atomic_projections() {
        use std::cell::Cell;
        struct CountProj<'a> {
            inner: &'a FuzzyKg,
            calls: Cell<usize>,
        }
        impl AtomicScorer for CountProj<'_> {
            fn num_entities(&self) -> usize {
                self.inner.num_entities()
            }
            fn project(&self, a: usize, r: usize) -> Vec<f32> {
                self.calls.set(self.calls.get() + 1);
                self.inner.project(a, r)
            }
        }
        let kg = taxonomy();
        let cfg = QueryConfig::default();
        let queries = vec![
            Query::anchor(3, 0),
            Query::anchor(3, 0).then(0),
            Query::intersection(vec![Query::anchor(3, 0), Query::anchor(4, 0)]),
        ];
        // Baseline: evaluate each query with its own fresh cache (no sharing).
        let mut baseline = 0usize;
        for q in &queries {
            let s = CountProj {
                inner: &kg,
                calls: Cell::new(0),
            };
            let _ = answer_query::<Godel>(&s, q, &cfg);
            baseline += s.calls.get();
        }
        // Batched: one shared cache across all three queries.
        let scorer = CountProj {
            inner: &kg,
            calls: Cell::new(0),
        };
        let _ = answer_queries::<Godel>(&scorer, &queries, &cfg);
        let batched = scorer.calls.get();
        // The batch must strictly reduce atomic-leaf recomputation: two of the
        // queries share `anchor(3, 0)`, which is computed once instead of
        // twice. Other intermediate projections are equal between the two
        // paths, so the batch total must be lower.
        assert!(
            batched < baseline,
            "batched {batched} should be < per-query baseline {baseline}"
        );
    }
}
