//! The query algebra: a compositional [`Query`] evaluated over an
//! [`AtomicScorer`] in a chosen [`Truth`] algebra.
//!
//! This mirrors `tranz::query` (CQD-Beam, Arakelyan et al. 2021) but abstracts
//! the model behind [`AtomicScorer`], so the same engine answers queries over
//! point embeddings (tranz), region embeddings (subsume), or any other source
//! of one-hop membership degrees. The geometry lives entirely behind
//! `project`; the engine only ever sees degrees in `[0, 1]`.

use crate::truth::Truth;

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

/// A complex logical query as a computation DAG over atomic projections.
///
/// Build with the constructors ([`Query::anchor`], [`Query::then`], etc.) and
/// evaluate with [`answer_query`] / [`answer_query_topk`]. The connectives
/// realize existential positive first-order logic plus negation and the Heyting
/// implication, each interpreted through the chosen [`Truth`] algebra.
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
    eval::<T>(scorer, query, config, n)
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

fn eval<T: Truth>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
    n: usize,
) -> Vec<f32> {
    match query {
        Query::Anchor { entity, relation } => {
            let mut s = scorer.project(*entity, *relation);
            s.resize(n, 0.0);
            s
        }
        Query::Project { inner, relation } => {
            let inner_scores = eval::<T>(scorer, inner, config, n);
            project::<T>(scorer, &inner_scores, *relation, config, n)
        }
        Query::Intersection { branches } => {
            // ⋀ branches via the t-norm, starting from ⊤.
            let mut acc = vec![T::top(); n];
            for branch in branches {
                let s = eval::<T>(scorer, branch, config, n);
                for (a, b) in acc.iter_mut().zip(s.iter()) {
                    *a = T::and(*a, *b);
                }
            }
            acc
        }
        Query::Union { branches } => {
            // ⋁ branches via the t-conorm, starting from ⊥.
            let mut acc = vec![T::bot(); n];
            for branch in branches {
                let s = eval::<T>(scorer, branch, config, n);
                for (a, b) in acc.iter_mut().zip(s.iter()) {
                    *a = T::or(*a, *b);
                }
            }
            acc
        }
        Query::Negation { inner } => {
            let mut s = eval::<T>(scorer, inner, config, n);
            for x in &mut s {
                *x = T::neg(*x);
            }
            s
        }
        Query::Implication {
            premise,
            conclusion,
        } => {
            let p = eval::<T>(scorer, premise, config, n);
            let c = eval::<T>(scorer, conclusion, config, n);
            p.iter()
                .zip(c.iter())
                .map(|(&pi, &ci)| T::residuum(pi, ci))
                .collect()
        }
    }
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
    let beam = top_k_descending(inner_scores, config.beam_k);
    let mut out = vec![0.0_f32; n];
    for &(v, v_score) in &beam {
        if v_score <= 0.0 {
            continue;
        }
        let tails = scorer.project(v, relation);
        for (t, &tail) in tails.iter().enumerate().take(n) {
            let combined = T::and(v_score, tail);
            if combined > out[t] {
                out[t] = combined;
            }
        }
    }
    out
}

fn top_k_descending(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
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
}
