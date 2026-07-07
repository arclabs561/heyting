//! Conformal answer sets: coverage-guaranteed query answering.
//!
//! A fuzzy degree ranks answers but does not say *how many* of the top
//! entities to trust. Split conformal prediction converts any scorer's
//! degrees into an answer **set** with a finite-sample guarantee: calibrate
//! on `n` held-out `(query, true answer)` pairs, and the set for a fresh
//! exchangeable query contains its true answer with probability at least
//! `1 - alpha` — no assumptions on the scorer, the geometry, or the training
//! procedure. This is the conformalized-answer-set construction for
//! knowledge-graph embeddings of Zhu et al. (NAACL 2025), applied to
//! [`answer_query`] degrees, so it wraps every [`AtomicScorer`] and every
//! [`Truth`] algebra uniformly.
//!
//! Mechanics: the nonconformity of a true answer is `1 - degree`; [`calibrate`]
//! takes the `ceil((n + 1) * (1 - alpha))`-th smallest calibration
//! nonconformity as the threshold `q̂`; [`answer_set`] then returns every
//! entity with `degree >= 1 - q̂`. When the rank exceeds `n` (too few
//! calibration examples for the requested confidence), the threshold is
//! conservative and the set is all entities — a correct, honest answer, not
//! an error.
//!
//! The guarantee is **marginal** (on average over exchangeable queries), not
//! per-query or per-relation. Predicate-conditional calibration (a separate
//! threshold per relation, Zhu et al., Findings ACL 2025) is a client-side
//! refinement: call [`calibrate`] once per relation with that relation's
//! examples.
//!
//! Off-seam readouts: [`calibrate_scores`] and [`answer_set_from_degrees`] are
//! the scorer-agnostic core, taking raw nonconformity / degree vectors. A
//! readout that cannot be expressed as one [`Query`] over the [`AtomicScorer`]
//! seam (a conjunctive least-common-ancestor score formed geometrically from two
//! anchors, for instance) is still conformalizable: compute its per-query
//! degrees yourself and feed them in. [`calibrate`] and [`answer_set`] are the
//! seam-bound convenience wrappers over that core.

use crate::query::{answer_query, AtomicScorer, Query, QueryConfig};
use crate::truth::Truth;

/// A calibrated nonconformity threshold from [`calibrate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalThreshold {
    /// The calibration quantile `q̂`; `f32::INFINITY` when the calibration
    /// set is too small for the requested confidence (full-set fallback).
    pub qhat: f32,
    /// The miscoverage level the threshold was calibrated for.
    pub alpha: f32,
    /// Number of calibration examples.
    pub n_calibration: usize,
}

/// Input problems [`calibrate`] rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformalError {
    /// `alpha` must be strictly inside `(0, 1)`.
    InvalidAlpha,
    /// The calibration set is empty.
    NoCalibrationExamples,
    /// A calibration answer id is out of range for the scorer.
    AnswerOutOfRange,
}

impl std::fmt::Display for ConformalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha => write!(f, "alpha must be in (0, 1)"),
            Self::NoCalibrationExamples => write!(f, "calibration set is empty"),
            Self::AnswerOutOfRange => write!(f, "calibration answer id out of range"),
        }
    }
}

impl std::error::Error for ConformalError {}

/// Calibrate a nonconformity threshold on `(query, true answer)` pairs.
///
/// Evaluates each query with [`answer_query`] in the algebra `T` and records
/// the true answer's nonconformity `1 - degree`. The returned threshold's
/// [`answer_set`] then carries the split-conformal guarantee: for a fresh
/// query exchangeable with the calibration pairs,
/// `P(true answer ∈ set) >= 1 - alpha`.
///
/// # Errors
///
/// [`ConformalError::InvalidAlpha`] unless `0 < alpha < 1`;
/// [`ConformalError::NoCalibrationExamples`] on an empty slice;
/// [`ConformalError::AnswerOutOfRange`] if an answer id is not an entity.
pub fn calibrate<T: Truth>(
    scorer: &dyn AtomicScorer,
    examples: &[(Query, usize)],
    config: &QueryConfig,
    alpha: f32,
) -> Result<ConformalThreshold, ConformalError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(ConformalError::InvalidAlpha);
    }
    if examples.is_empty() {
        return Err(ConformalError::NoCalibrationExamples);
    }
    let n_entities = scorer.num_entities();
    let mut nonconformities = Vec::with_capacity(examples.len());
    for (query, answer) in examples {
        if *answer >= n_entities {
            return Err(ConformalError::AnswerOutOfRange);
        }
        let degrees = answer_query::<T>(scorer, query, config);
        nonconformities.push(1.0 - degrees[*answer]);
    }
    calibrate_scores(&nonconformities, alpha)
}

/// Calibrate a threshold directly from precomputed nonconformity scores.
///
/// The scorer-agnostic core of [`calibrate`]: given each calibration example's
/// nonconformity (higher = worse fit, conventionally `1 - degree` but any
/// exchangeable score works), return the finite-sample conformal threshold, the
/// `ceil((n + 1) * (1 - alpha))`-th smallest nonconformity (clamped to
/// `[0, 1]`), or `INFINITY` when the rank exceeds `n` (conservative full-set
/// fallback).
///
/// Use this to conformalize a readout that does *not* fit the atomic
/// [`AtomicScorer`] `project(anchor, relation)` seam that [`calibrate`] is built
/// on: a conjunctive score formed geometrically from several anchors (an LCA
/// join, an intersection materialized off-seam) cannot be expressed as one
/// [`Query`], so compute its per-example nonconformities yourself and calibrate
/// here, then form sets with [`answer_set_from_degrees`].
///
/// # Errors
///
/// [`ConformalError::InvalidAlpha`] unless `0 < alpha < 1`;
/// [`ConformalError::NoCalibrationExamples`] on an empty slice.
pub fn calibrate_scores(
    nonconformities: &[f32],
    alpha: f32,
) -> Result<ConformalThreshold, ConformalError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(ConformalError::InvalidAlpha);
    }
    if nonconformities.is_empty() {
        return Err(ConformalError::NoCalibrationExamples);
    }
    let mut scores: Vec<f32> = nonconformities.iter().map(|s| s.clamp(0.0, 1.0)).collect();
    scores.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = scores.len();
    // Finite-sample conformal quantile: the ceil((n + 1)(1 - alpha))-th
    // smallest score. Rank > n means the guarantee needs more examples than
    // provided; fall back to the conservative full set.
    let rank = ((n as f64 + 1.0) * (1.0 - alpha as f64)).ceil() as usize;
    let qhat = if rank > n {
        f32::INFINITY
    } else {
        scores[rank - 1]
    };
    Ok(ConformalThreshold {
        qhat,
        alpha,
        n_calibration: n,
    })
}

/// The conformal answer set: every entity whose degree reaches `1 - q̂`,
/// with its degree, best first.
pub fn answer_set<T: Truth>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
    threshold: &ConformalThreshold,
) -> Vec<(usize, f32)> {
    let degrees = answer_query::<T>(scorer, query, config);
    answer_set_from_degrees(&degrees, threshold)
}

/// The conformal answer set from a precomputed per-entity degree vector.
///
/// The scorer-agnostic core of [`answer_set`]: every entity whose degree reaches
/// `1 - q̂` (from [`calibrate_scores`] or [`calibrate`] over the same readout),
/// best first, ties broken by id. `degrees[i]` is entity `i`'s degree; the
/// off-seam companion to [`calibrate_scores`] for conformalizing a readout the
/// atomic [`AtomicScorer`] seam cannot express.
pub fn answer_set_from_degrees(
    degrees: &[f32],
    threshold: &ConformalThreshold,
) -> Vec<(usize, f32)> {
    let cutoff = 1.0 - threshold.qhat; // -inf when qhat is inf: everything.
    let mut set: Vec<(usize, f32)> = degrees
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, d)| *d >= cutoff)
        .collect();
    set.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    set
}

/// The conformal answer set from a sparse scored candidate pool.
///
/// This is the candidate-pool companion to [`answer_set_from_degrees`]. It
/// applies the same `degree >= 1 - q̂` cutoff, but only over candidates the
/// caller supplied. When `q̂` is infinite, the conservative fallback is the full
/// candidate pool, not every possible entity. If a candidate id appears more
/// than once, the highest supplied degree is retained.
pub fn answer_set_from_scored_pool(
    scored: &[(usize, f32)],
    threshold: &ConformalThreshold,
) -> Vec<(usize, f32)> {
    let cutoff = 1.0 - threshold.qhat;
    let mut best_by_id = std::collections::BTreeMap::new();
    for &(entity, degree) in scored {
        if degree >= cutoff {
            best_by_id
                .entry(entity)
                .and_modify(|best| {
                    if degree > *best {
                        *best = degree;
                    }
                })
                .or_insert(degree);
        }
    }
    let mut set: Vec<(usize, f32)> = best_by_id.into_iter().collect();
    set.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    set
}

/// Fraction of `(query, true answer)` pairs whose answer set contains the
/// true answer. On exchangeable held-out pairs this should be at least
/// `1 - alpha` up to finite-sample noise.
pub fn empirical_coverage<T: Truth>(
    scorer: &dyn AtomicScorer,
    tests: &[(Query, usize)],
    config: &QueryConfig,
    threshold: &ConformalThreshold,
) -> f32 {
    if tests.is_empty() {
        return 0.0;
    }
    let hits = tests
        .iter()
        .filter(|(query, answer)| {
            answer_set::<T>(scorer, query, config, threshold)
                .iter()
                .any(|(e, _)| e == answer)
        })
        .count();
    hits as f32 / tests.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::FuzzyKg;
    use crate::truth::Godel;

    /// A graph whose 1p degrees are exactly the edge weights, so calibration
    /// nonconformities are hand-computable.
    fn kg() -> FuzzyKg {
        let mut kg = FuzzyKg::new(6);
        kg.add_edge(0, 0, 1, 0.9); // s = 0.1
        kg.add_edge(2, 0, 3, 0.8); // s = 0.2
        kg.add_edge(4, 0, 5, 0.7); // s = 0.3
        kg.add_edge(1, 0, 3, 0.6); // s = 0.4
        kg
    }

    fn calibration() -> Vec<(Query, usize)> {
        vec![
            (Query::anchor(0, 0), 1),
            (Query::anchor(2, 0), 3),
            (Query::anchor(4, 0), 5),
            (Query::anchor(1, 0), 3),
        ]
    }

    /// Hand-computed conformal quantiles over nonconformities
    /// {0.1, 0.2, 0.3, 0.4} (n = 4): rank = ceil(5 * (1 - alpha)).
    #[test]
    fn quantile_matches_hand_computation() {
        let kg = kg();
        let cfg = QueryConfig::default();

        // alpha = 0.25: rank ceil(3.75) = 4 -> qhat = 0.4.
        let t = calibrate::<Godel>(&kg, &calibration(), &cfg, 0.25).unwrap();
        assert!((t.qhat - 0.4).abs() < 1e-6, "qhat {}", t.qhat);

        // alpha = 0.5: rank ceil(2.5) = 3 -> qhat = 0.3.
        let t = calibrate::<Godel>(&kg, &calibration(), &cfg, 0.5).unwrap();
        assert!((t.qhat - 0.3).abs() < 1e-6, "qhat {}", t.qhat);

        // alpha = 0.01: rank ceil(4.95) = 5 > 4 -> conservative full set.
        let t = calibrate::<Godel>(&kg, &calibration(), &cfg, 0.01).unwrap();
        assert!(t.qhat.is_infinite());
    }

    /// With qhat = 0.3 the answer-set cutoff is degree >= 0.7: the 0.9, 0.8,
    /// 0.7 answers are in their sets, the 0.6 answer is not. Coverage over
    /// the four calibration pairs is exactly 3/4 = 1 - alpha at alpha = 0.5
    /// (the guarantee holding with equality on this construction).
    #[test]
    fn answer_sets_apply_the_cutoff() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let t = calibrate::<Godel>(&kg, &calibration(), &cfg, 0.5).unwrap();

        let set = answer_set::<Godel>(&kg, &Query::anchor(0, 0), &cfg, &t);
        assert_eq!(set.first().map(|(e, _)| *e), Some(1));
        let set = answer_set::<Godel>(&kg, &Query::anchor(1, 0), &cfg, &t);
        assert!(
            !set.iter().any(|(e, _)| *e == 3),
            "0.6 < cutoff 0.7: {set:?}"
        );

        let cov = empirical_coverage::<Godel>(&kg, &calibration(), &cfg, &t);
        assert!((cov - 0.75).abs() < 1e-6, "coverage {cov}");
    }

    /// The conservative fallback covers everything.
    #[test]
    fn infinite_threshold_returns_all_entities() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let t = calibrate::<Godel>(&kg, &calibration(), &cfg, 0.01).unwrap();
        let set = answer_set::<Godel>(&kg, &Query::anchor(0, 0), &cfg, &t);
        assert_eq!(set.len(), kg.num_entities());
        let cov = empirical_coverage::<Godel>(&kg, &calibration(), &cfg, &t);
        assert!((cov - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_inputs() {
        let kg = kg();
        let cfg = QueryConfig::default();
        assert_eq!(
            calibrate::<Godel>(&kg, &calibration(), &cfg, 0.0).unwrap_err(),
            ConformalError::InvalidAlpha
        );
        assert_eq!(
            calibrate::<Godel>(&kg, &calibration(), &cfg, 1.0).unwrap_err(),
            ConformalError::InvalidAlpha
        );
        assert_eq!(
            calibrate::<Godel>(&kg, &[], &cfg, 0.1).unwrap_err(),
            ConformalError::NoCalibrationExamples
        );
        assert_eq!(
            calibrate::<Godel>(&kg, &[(Query::anchor(0, 0), 99)], &cfg, 0.1).unwrap_err(),
            ConformalError::AnswerOutOfRange
        );
    }

    /// The score-vector core reproduces the seam quantiles directly from raw
    /// nonconformities {0.1, 0.2, 0.3, 0.4}, no scorer involved.
    #[test]
    fn calibrate_scores_matches_hand_computation() {
        let nonconf = [0.1f32, 0.2, 0.3, 0.4];
        // alpha = 0.25: rank ceil(3.75) = 4 -> qhat = 0.4.
        assert!((calibrate_scores(&nonconf, 0.25).unwrap().qhat - 0.4).abs() < 1e-6);
        // alpha = 0.5: rank ceil(2.5) = 3 -> qhat = 0.3.
        assert!((calibrate_scores(&nonconf, 0.5).unwrap().qhat - 0.3).abs() < 1e-6);
        // alpha = 0.01: rank 5 > 4 -> conservative full set.
        assert!(calibrate_scores(&nonconf, 0.01).unwrap().qhat.is_infinite());
        assert_eq!(
            calibrate_scores(&[], 0.1).unwrap_err(),
            ConformalError::NoCalibrationExamples
        );
        assert_eq!(
            calibrate_scores(&nonconf, 1.0).unwrap_err(),
            ConformalError::InvalidAlpha
        );
    }

    /// The seam-bound calibrate is exactly its score core over `1 - degree`:
    /// the kg's calibration degrees 0.9/0.8/0.7/0.6 give nonconformities
    /// 0.1/0.2/0.3/0.4, so both paths must agree, proving the refactor
    /// preserves behaviour.
    #[test]
    fn seam_calibrate_delegates_to_score_core() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let nonconf = [0.1f32, 0.2, 0.3, 0.4];
        for &alpha in &[0.25f32, 0.5, 0.01] {
            let seam = calibrate::<Godel>(&kg, &calibration(), &cfg, alpha).unwrap();
            let core = calibrate_scores(&nonconf, alpha).unwrap();
            assert_eq!(seam.qhat.is_infinite(), core.qhat.is_infinite());
            if seam.qhat.is_finite() {
                assert!((seam.qhat - core.qhat).abs() < 1e-6, "alpha {alpha}");
            }
        }
    }

    /// answer_set_from_degrees applies the `1 - qhat` cutoff to a raw degree
    /// vector, best first with ties by id.
    #[test]
    fn answer_set_from_degrees_applies_cutoff() {
        // qhat 0.3 -> cutoff degree >= 0.7.
        let thr = calibrate_scores(&[0.1, 0.2, 0.3, 0.4], 0.5).unwrap();
        let degrees = [0.9f32, 0.75, 0.7, 0.6, 0.95];
        let ids: Vec<usize> = answer_set_from_degrees(&degrees, &thr)
            .iter()
            .map(|(e, _)| *e)
            .collect();
        // 0.95, 0.9, 0.75, 0.7 pass (best first); 0.6 does not.
        assert_eq!(ids, vec![4, 0, 1, 2]);
    }

    /// answer_set_from_scored_pool applies the conformal cutoff to only the
    /// candidate ids supplied by the caller.
    #[test]
    fn answer_set_from_scored_pool_applies_cutoff_to_sparse_candidates() {
        let thr = calibrate_scores(&[0.1, 0.2, 0.3, 0.4], 0.5).unwrap();
        let scored = [(10usize, 0.9f32), (5, 0.65), (7, 0.75), (10, 0.85)];
        let set = answer_set_from_scored_pool(&scored, &thr);
        assert_eq!(set, vec![(10, 0.9), (7, 0.75)]);
    }

    /// With too little calibration data, the sparse-pool fallback includes the
    /// full candidate pool rather than pretending unscored entities exist.
    #[test]
    fn answer_set_from_scored_pool_full_fallback_stays_inside_pool() {
        let thr = ConformalThreshold {
            qhat: f32::INFINITY,
            alpha: 0.1,
            n_calibration: 1,
        };
        let scored = [(9usize, 0.1f32), (3, 0.8)];
        let set = answer_set_from_scored_pool(&scored, &thr);
        assert_eq!(set, vec![(3, 0.8), (9, 0.1)]);
    }
}
