//! Query-answering evaluation with the field-standard easy/hard answer split.
//!
//! The CLQA literature (Query2Box, BetaE, and everything benchmarked against
//! them) evaluates a query in two answer classes:
//!
//! - **easy**: answers reachable by traversing edges the model was trained on.
//!   A graph lookup finds these; they say nothing about generalization.
//! - **hard**: answers that hold in the full graph but require predicting at
//!   least one missing edge. Metrics are reported on these, with easy answers
//!   (and the other hard answers) filtered out of the ranking.
//!
//! [`split_answers`] computes the split by evaluating the query *crisply*
//! (exhaustive traversal, no beam truncation) over a train scorer and a full
//! scorer; [`hard_answer_metrics`] then scores a fuzzy answer vector against
//! it in the filtered setting.
//!
//! # Query-shape names
//!
//! The standard taxonomy names tree-form query shapes; the [`Query`]
//! constructors build those shapes:
//!
//! | Name | Shape | Construction |
//! |---|---|---|
//! | `1p` | one hop | `Query::anchor(e, r)` |
//! | `2p`, `3p` | projection chain | `.then(r2)`, `.then(r2).then(r3)` |
//! | `2i`, `3i` | intersection | `Query::intersection(vec![...])` |
//! | `pi` | project then intersect | `intersection(vec![anchor(e,r1).then(r2), anchor(e2,r3)])` |
//! | `ip` | intersect then project | `intersection(vec![...]).then(r)` |
//! | `2u` | union | `Query::union(vec![...])` |
//! | `up` | union then project | `union(vec![...]).then(r)` |
//! | `2in`, `3in`, `inp`, `pin`, `pni` | negation variants | `.negate()` inside the shapes above |

use std::collections::BTreeSet;

use crate::query::{answer_query, AtomicScorer, Query, QueryConfig};
use crate::truth::Godel;
use crate::Truth;

/// The easy/hard answer split for one query.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryAnswers {
    /// Answers derivable from the train graph by traversal.
    pub easy: BTreeSet<usize>,
    /// Answers that hold in the full graph but not by train-graph traversal.
    pub hard: BTreeSet<usize>,
}

/// Ranking metrics over the hard answers of one query (filtered setting).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct QueryMetrics {
    /// Mean reciprocal rank over hard answers.
    pub mrr: f32,
    /// Fraction of hard answers ranked first.
    pub hits1: f32,
    /// Fraction of hard answers ranked in the top 3.
    pub hits3: f32,
    /// Fraction of hard answers ranked in the top 10.
    pub hits10: f32,
    /// Number of hard answers scored.
    pub n_hard: usize,
}

/// Query answers plus optional cardinality metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QueryAnswerReport {
    /// Top answers sorted by degree descending, then entity id ascending.
    pub top_k: Vec<(usize, f32)>,
    /// Degree for every entity in scorer order.
    pub degrees: Vec<f32>,
    /// Optional predicted answer cardinality. Geometric materializers can
    /// populate this from region volume; plain degree scorers leave it absent.
    pub predicted_cardinality: Option<f32>,
}

/// Evaluate a query and return both the full degree vector and top-k answers.
pub fn answer_query_report<T: Truth>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
    k: usize,
) -> QueryAnswerReport {
    // Evaluate once; derive top-k from the same dense vector so we do not pay
    // for a second full `answer_query` just to rank it.
    let degrees = crate::query::answer_query::<T>(scorer, query, config);
    let top_k = crate::query::top_k_descending(&degrees, k);
    QueryAnswerReport {
        top_k,
        degrees,
        predicted_cardinality: None,
    }
}

/// Entities whose crisp query degree reaches `threshold`.
///
/// Evaluates with [`Godel`] (on `{0, 1}` edge weights every algebra coincides
/// with boolean logic) and an exhaustive beam (`beam_k = num_entities`), so
/// the result is set-theoretic traversal, not a beam approximation. Negation
/// is closed-world with respect to `scorer`'s edges, matching how the
/// benchmark suites generate negation queries. `threshold = 0.5` is the
/// natural choice for crisp graphs.
pub fn crisp_answers(scorer: &dyn AtomicScorer, query: &Query, threshold: f32) -> BTreeSet<usize> {
    let config = QueryConfig {
        beam_k: scorer.num_entities(),
    };
    answer_query::<Godel>(scorer, query, &config)
        .iter()
        .enumerate()
        .filter(|(_, &d)| d >= threshold)
        .map(|(e, _)| e)
        .collect()
}

/// Split a query's answers into easy (train-traversable) and hard (require a
/// missing edge) using crisp traversal over the two graphs.
///
/// `train` holds the edges a model was trained on; `full` additionally holds
/// the held-out (valid/test) edges. See [`crisp_answers`] for the traversal
/// semantics and the role of `threshold`.
pub fn split_answers(
    train: &dyn AtomicScorer,
    full: &dyn AtomicScorer,
    query: &Query,
    threshold: f32,
) -> QueryAnswers {
    let easy = crisp_answers(train, query, threshold);
    let mut hard = crisp_answers(full, query, threshold);
    hard.retain(|e| !easy.contains(e));
    QueryAnswers { easy, hard }
}

/// Score a fuzzy answer vector against the split, reporting on hard answers
/// only, in the filtered setting: when ranking one hard answer, the easy
/// answers and the *other* hard answers are removed from the candidate list,
/// so true answers do not compete with each other.
///
/// `scores` is an [`answer_query`] output (one degree per
/// entity). Ties rank pessimistically (a tied competitor counts as ahead).
/// Returns zeroed metrics with `n_hard = 0` when the query has no hard answer.
pub fn hard_answer_metrics(scores: &[f32], answers: &QueryAnswers) -> QueryMetrics {
    if answers.hard.is_empty() {
        return QueryMetrics::default();
    }
    let (mut mrr, mut h1, mut h3, mut h10) = (0.0_f64, 0usize, 0usize, 0usize);
    for &target in &answers.hard {
        // An entity past the end of `scores` has degree 0.0 by the engine's
        // own convention (`answer_query` pads with 0.0), not as an error
        // sentinel; the ranking below treats it as a bottom-scored answer.
        let target_score = scores.get(target).copied().unwrap_or(0.0);
        let mut rank = 1usize;
        for (e, &s) in scores.iter().enumerate() {
            if e == target || answers.easy.contains(&e) || answers.hard.contains(&e) {
                continue;
            }
            if s >= target_score {
                rank += 1;
            }
        }
        mrr += 1.0 / rank as f64;
        h1 += usize::from(rank <= 1);
        h3 += usize::from(rank <= 3);
        h10 += usize::from(rank <= 10);
    }
    let n = answers.hard.len();
    QueryMetrics {
        mrr: (mrr / n as f64) as f32,
        hits1: h1 as f32 / n as f32,
        hits3: h3 as f32 / n as f32,
        hits10: h10 as f32 / n as f32,
        n_hard: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::FuzzyKg;

    /// 0=animal 1=mammal 2=dog 3=cat 4=fish; relation 0 = is_a.
    /// Train graph misses cat->mammal; the full graph has it.
    fn graphs() -> (FuzzyKg, FuzzyKg) {
        let mut train = FuzzyKg::new(5);
        train.add_edge(2, 0, 1, 1.0);
        train.add_edge(1, 0, 0, 1.0);
        let mut full = train.clone();
        full.add_edge(3, 0, 1, 1.0);
        (train, full)
    }

    #[test]
    fn split_separates_traversable_from_predicted() {
        let (train, full) = graphs();
        // 1p: cat is_a ? — nothing traversable in train, mammal hard.
        let q = Query::anchor(3, 0);
        let a = split_answers(&train, &full, &q, 0.5);
        assert!(a.easy.is_empty());
        assert_eq!(a.hard, BTreeSet::from([1]));

        // 1p: dog is_a ? — mammal is easy, nothing hard.
        let q = Query::anchor(2, 0);
        let a = split_answers(&train, &full, &q, 0.5);
        assert_eq!(a.easy, BTreeSet::from([1]));
        assert!(a.hard.is_empty());
    }

    #[test]
    fn split_covers_projection_chains() {
        let (train, full) = graphs();
        // 2p: cat is_a ? is_a ? — animal requires the missing cat edge.
        let q = Query::anchor(3, 0).then(0);
        let a = split_answers(&train, &full, &q, 0.5);
        assert!(a.easy.is_empty());
        assert_eq!(a.hard, BTreeSet::from([0]));
    }

    #[test]
    fn metrics_filter_easy_and_other_hard_answers() {
        // Model scores: entity 1 (hard) outranked only by entity 4 and by
        // easy entity 0; filtering removes 0, so the rank is 2.
        let answers = QueryAnswers {
            easy: BTreeSet::from([0]),
            hard: BTreeSet::from([1]),
        };
        let scores = vec![0.9, 0.5, 0.1, 0.2, 0.7];
        let m = hard_answer_metrics(&scores, &answers);
        assert_eq!(m.n_hard, 1);
        assert!((m.mrr - 0.5).abs() < 1e-6);
        assert!((m.hits1 - 0.0).abs() < 1e-6);
        assert!((m.hits3 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn perfect_model_gets_mrr_one() {
        let (train, full) = graphs();
        let q = Query::anchor(3, 0);
        let answers = split_answers(&train, &full, &q, 0.5);
        // The full graph itself is the perfect scorer for this query.
        let scores = crate::answer_query::<Godel>(&full, &q, &QueryConfig::default());
        let m = hard_answer_metrics(&scores, &answers);
        assert_eq!(m.n_hard, 1);
        assert!((m.mrr - 1.0).abs() < 1e-6);
        assert!((m.hits1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_hard_answers_reports_zeroed_metrics() {
        let answers = QueryAnswers::default();
        let m = hard_answer_metrics(&[0.1, 0.2], &answers);
        assert_eq!(m.n_hard, 0);
        assert_eq!(m.mrr, 0.0);
    }

    #[test]
    fn answer_report_contains_degrees_and_topk() {
        let (_train, full) = graphs();
        let q = Query::anchor(3, 0);
        let report = answer_query_report::<Godel>(&full, &q, &QueryConfig::default(), 2);
        assert_eq!(report.degrees.len(), full.num_entities());
        assert_eq!(report.top_k.len(), 2);
        assert_eq!(report.predicted_cardinality, None);
    }
}
