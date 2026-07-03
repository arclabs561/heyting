//! Candidate-pruned query evaluation: the retrieval-backend seam.
//!
//! [`answer_query_topk`](crate::answer_query_topk) scores every entity at
//! every DAG node. When an index can propose a small candidate set for each
//! atomic hop — a k-NN index over point embeddings, a region index answering
//! membership — most of that work is provably wasted: entities outside every
//! hop's candidate set can only carry degree `0`, and `0` is absorbing for
//! conjunction and neutral for disjunction in every [`Truth`] algebra. The
//! pruned evaluator exploits exactly that: it tracks only the entities some
//! atomic hop proposed, calling
//! [`AtomicScorer::project_subset`] instead of the dense
//! [`AtomicScorer::project`].
//!
//! [`CandidateSource`] is the seam: implement it with whatever serving index
//! fits the scorer's geometry. Recall is bounded by the candidate source (a
//! candidate set that misses a true answer loses it), so a source should
//! over-propose the way an ANN index over-retrieves. Returning `None` for a
//! hop means "no pruning here" and that hop falls back to dense scoring.
//!
//! # Scope: existential-positive queries
//!
//! Pruning is sound for anchor / projection / intersection / union queries
//! (the EPFO fragment). Negation and implication invert degrees, so entities
//! *outside* every candidate set become answers; a query containing either
//! connective is evaluated densely by
//! [`answer_query_topk_pruned`] (identical results, no pruning benefit).
//!
//! [`Truth`]: crate::Truth

use std::collections::HashMap;

use crate::query::{AtomicScorer, Query, QueryConfig};
use crate::truth::Truth;

/// Proposes candidate answers for atomic hops; the serving-index seam.
pub trait CandidateSource {
    /// Candidate answer entities for the atomic hop `(anchor, relation, ?)`,
    /// or `None` to score that hop densely. Order and duplicates are
    /// irrelevant; a missing true answer is a recall loss, so over-propose.
    fn candidates(&self, anchor: usize, relation: usize) -> Option<Vec<usize>>;
}

/// Every edge's tails are its hop's candidates: exact for [`crate::FuzzyKg`],
/// where non-edges have degree `0` by construction.
impl CandidateSource for crate::FuzzyKg {
    fn candidates(&self, anchor: usize, relation: usize) -> Option<Vec<usize>> {
        Some(self.tails(anchor, relation))
    }
}

/// Answer `query`, pruning each atomic hop to the candidates proposed by
/// `source`, and return the top-`k` `(entity, degree)` pairs, best first.
///
/// Equivalent to [`answer_query_topk`](crate::answer_query_topk) whenever the
/// candidate sets cover the true nonzero support of each hop; recall degrades
/// gracefully with candidate coverage otherwise. Queries containing negation
/// or implication are evaluated densely (see the module doc).
pub fn answer_query_topk_pruned<T: Truth>(
    scorer: &dyn AtomicScorer,
    source: &dyn CandidateSource,
    query: &Query,
    config: &QueryConfig,
    k: usize,
) -> Vec<(usize, f32)> {
    if has_inverting_connective(query) {
        return crate::query::answer_query_topk::<T>(scorer, query, config, k);
    }
    let sparse = eval_sparse::<T>(scorer, source, query, config);
    let mut pairs: Vec<(usize, f32)> = sparse.into_iter().collect();
    pairs.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    pairs.truncate(k);
    pairs
}

fn has_inverting_connective(query: &Query) -> bool {
    match query {
        Query::Anchor { .. } => false,
        Query::Project { inner, .. } => has_inverting_connective(inner),
        Query::Intersection { branches } | Query::Union { branches } => {
            branches.iter().any(has_inverting_connective)
        }
        Query::Negation { .. } | Query::Implication { .. } => true,
    }
}

/// Sparse degrees: entities absent from the map carry degree `0`.
fn eval_sparse<T: Truth>(
    scorer: &dyn AtomicScorer,
    source: &dyn CandidateSource,
    query: &Query,
    config: &QueryConfig,
) -> HashMap<usize, f32> {
    match query {
        Query::Anchor { entity, relation } => hop(scorer, source, *entity, *relation),
        Query::Project { inner, relation } => {
            let inner_scores = eval_sparse::<T>(scorer, source, inner, config);
            let pairs: Vec<(usize, f32)> = inner_scores.into_iter().collect();
            let beam = top_k_descending_sparse(&pairs, config.beam_k);
            let mut out: HashMap<usize, f32> = HashMap::new();
            for &(v, v_score) in &beam {
                if v_score <= 0.0 {
                    continue;
                }
                for (t, t_score) in hop(scorer, source, v, *relation) {
                    // ∃V: conjoin the intermediate's degree, join (max) over
                    // intermediates — same combination as the dense engine.
                    let d = T::and(v_score, t_score);
                    let e = out.entry(t).or_insert(0.0);
                    if d > *e {
                        *e = d;
                    }
                }
            }
            out
        }
        Query::Intersection { branches } => {
            // Absent = 0 and T::and(_, 0) = 0 in every algebra, so the result
            // support is the intersection of branch supports.
            let mut iter = branches.iter();
            let Some(first) = iter.next() else {
                return HashMap::new();
            };
            let mut acc = eval_sparse::<T>(scorer, source, first, config);
            for branch in iter {
                let s = eval_sparse::<T>(scorer, source, branch, config);
                acc = acc
                    .into_iter()
                    .filter_map(|(e, a)| s.get(&e).map(|&b| (e, T::and(a, b))))
                    .collect();
            }
            acc
        }
        Query::Union { branches } => {
            // Absent = 0 and T::or(a, 0) = a, so the union of supports.
            let mut acc: HashMap<usize, f32> = HashMap::new();
            for branch in branches {
                for (e, b) in eval_sparse::<T>(scorer, source, branch, config) {
                    let a = acc.entry(e).or_insert(T::bot());
                    *a = T::or(*a, b);
                }
            }
            acc
        }
        // Unreachable behind `has_inverting_connective`, but keep it correct:
        // evaluate densely and keep the nonzero entries.
        Query::Negation { .. } | Query::Implication { .. } => {
            crate::query::answer_query::<T>(scorer, query, config)
                .into_iter()
                .enumerate()
                .filter(|(_, d)| *d > 0.0)
                .collect()
        }
    }
}

fn hop(
    scorer: &dyn AtomicScorer,
    source: &dyn CandidateSource,
    anchor: usize,
    relation: usize,
) -> HashMap<usize, f32> {
    match source.candidates(anchor, relation) {
        Some(mut cand) => {
            cand.sort_unstable();
            cand.dedup();
            let degrees = scorer.project_subset(anchor, relation, &cand);
            cand.into_iter()
                .zip(degrees)
                .filter(|(_, d)| *d > 0.0)
                .collect()
        }
        None => scorer
            .project(anchor, relation)
            .into_iter()
            .enumerate()
            .filter(|(_, d)| *d > 0.0)
            .collect(),
    }
}

/// Top-`k` of sparse `(entity, degree)` pairs, best first, ties by entity id.
fn top_k_descending_sparse(pairs: &[(usize, f32)], k: usize) -> Vec<(usize, f32)> {
    let mut v = pairs.to_vec();
    v.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    v.truncate(k);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::FuzzyKg;
    use crate::query::answer_query_topk;
    use crate::truth::{Godel, Lukasiewicz, Product};

    /// 0=animal 1=mammal 2=dog 3=cat 4=fish; 0=is_a, 1=eats.
    fn kg() -> FuzzyKg {
        let mut kg = FuzzyKg::new(5);
        kg.add_edge(2, 0, 1, 0.9); // dog is_a mammal
        kg.add_edge(3, 0, 1, 0.8); // cat is_a mammal
        kg.add_edge(1, 0, 0, 1.0); // mammal is_a animal
        kg.add_edge(3, 1, 4, 0.7); // cat eats fish
        kg.add_edge(2, 1, 4, 0.4); // dog eats fish
        kg
    }

    /// Dense top-k pads with zero-degree entities (it sorts all n and
    /// truncates); pruned only ever surfaces positive degrees. Compare the
    /// positive entries under one canonical order (degree desc, id asc).
    fn assert_same_topk(dense: &[(usize, f32)], pruned: &[(usize, f32)]) {
        let canon = |xs: &[(usize, f32)]| {
            let mut v: Vec<(usize, f32)> = xs.iter().copied().filter(|(_, d)| *d > 0.0).collect();
            v.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            v
        };
        let (dense, pruned) = (canon(dense), canon(pruned));
        assert_eq!(dense.len(), pruned.len(), "{dense:?} vs {pruned:?}");
        for ((de, dd), (pe, pd)) in dense.iter().zip(pruned.iter()) {
            assert_eq!(de, pe, "{dense:?} vs {pruned:?}");
            assert!((dd - pd).abs() < 1e-6, "degree {dd} vs {pd}");
        }
    }

    /// FuzzyKg's own edges are an exact candidate source, so pruned == dense
    /// across the EPFO shapes and all three algebras.
    #[test]
    fn pruned_matches_dense_on_epfo_shapes() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let queries = [
            Query::anchor(2, 0),                                                 // 1p
            Query::anchor(3, 0).then(0),                                         // 2p
            Query::intersection(vec![Query::anchor(2, 0), Query::anchor(3, 0)]), // 2i
            Query::union(vec![Query::anchor(2, 1), Query::anchor(3, 1)]),        // 2u
            Query::intersection(vec![
                Query::anchor(2, 0).then(0),
                Query::anchor(3, 0).then(0),
            ]), // pi-shaped
            Query::union(vec![Query::anchor(2, 0), Query::anchor(3, 0)]).then(0), // up
        ];
        for q in &queries {
            assert_same_topk(
                &answer_query_topk::<Godel>(&kg, q, &cfg, 5),
                &answer_query_topk_pruned::<Godel>(&kg, &kg, q, &cfg, 5),
            );
            assert_same_topk(
                &answer_query_topk::<Product>(&kg, q, &cfg, 5),
                &answer_query_topk_pruned::<Product>(&kg, &kg, q, &cfg, 5),
            );
            assert_same_topk(
                &answer_query_topk::<Lukasiewicz>(&kg, q, &cfg, 5),
                &answer_query_topk_pruned::<Lukasiewicz>(&kg, &kg, q, &cfg, 5),
            );
        }
    }

    /// Negation-bearing queries take the dense path and still match.
    #[test]
    fn negation_falls_back_to_dense() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let q = Query::intersection(vec![Query::anchor(2, 0), Query::anchor(3, 0).negate()]);
        assert_same_topk(
            &answer_query_topk::<Lukasiewicz>(&kg, &q, &cfg, 5),
            &answer_query_topk_pruned::<Lukasiewicz>(&kg, &kg, &q, &cfg, 5),
        );
    }

    /// A candidate source that misses a true answer loses it: recall is the
    /// source's responsibility, which is the documented contract.
    #[test]
    fn missing_candidate_is_a_recall_loss() {
        struct Blind;
        impl CandidateSource for Blind {
            fn candidates(&self, _: usize, _: usize) -> Option<Vec<usize>> {
                Some(vec![]) // proposes nothing
            }
        }
        let kg = kg();
        let out = answer_query_topk_pruned::<Godel>(
            &kg,
            &Blind,
            &Query::anchor(2, 0),
            &QueryConfig::default(),
            5,
        );
        assert!(out.is_empty());
    }

    /// `None` from the source means dense scoring for that hop.
    #[test]
    fn none_means_no_pruning() {
        struct NoOpinion;
        impl CandidateSource for NoOpinion {
            fn candidates(&self, _: usize, _: usize) -> Option<Vec<usize>> {
                None
            }
        }
        let kg = kg();
        let cfg = QueryConfig::default();
        let q = Query::anchor(3, 0).then(0);
        assert_same_topk(
            &answer_query_topk::<Godel>(&kg, &q, &cfg, 5),
            &answer_query_topk_pruned::<Godel>(&kg, &NoOpinion, &q, &cfg, 5),
        );
    }
}
