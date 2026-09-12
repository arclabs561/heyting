//! Split-conformal answer sets for query answering.
//!
//! A fuzzy degree ranks answers but does not say *how many* of the top
//! entities to trust. Split conformal prediction converts any scorer's
//! degrees into an answer **set** with a finite-sample guarantee: calibrate
//! on `n` held-out `(query, true answer)` pairs, and the set for a fresh
//! exchangeable query contains its true answer with probability at least
//! `1 - alpha`. Keep the scorer fixed independently of calibration.
//! This is the conformalized-answer-set construction for
//! knowledge-graph embeddings of Zhu et al. (NAACL 2025), applied to
//! [`answer_query`] degrees, so it wraps every [`AtomicScorer`] and every
//! [`Truth`] algebra uniformly.
//!
//! Mechanics: the nonconformity of a true answer is `1 - degree`; [`calibrate`]
//! takes the `ceil((n + 1) * (1 - alpha))`-th smallest calibration
//! nonconformity as the threshold `q̂`; [`answer_set`] then returns every
//! entity with `1 - degree <= q̂`. When the rank exceeds `n` (too few
//! calibration examples for the requested confidence), the threshold is
//! conservative and the set is all entities.
//!
//! Keep the scorer and score definition fixed independently of calibration
//! examples. Calibration and future scores must be exchangeable.
//!
//! The guarantee is **marginal** (on average over exchangeable queries), not
//! per-query or per-relation. Predicate-conditional calibration (a separate
//! threshold per relation, Zhu et al., Findings ACL 2025) is a client-side
//! refinement: call [`calibrate`] once per relation with that relation's
//! examples.
//!
//! [`calibrate_scores`] accepts raw nonconformity scores.
//! [`answer_set_from_degrees`] constructs sets for `1 - degree` scores;
//! callers using other scores construct sets on their own score scale.
//! [`calibrate`] and [`answer_set`] evaluate a [`Query`] through an
//! [`AtomicScorer`] before calling these helpers.

use crate::query::{answer_query, AtomicScorer, Query, QueryConfig};
use crate::truth::Truth;
use statskit::conformal::{
    calibrate_in_place, ConformalError as StatskitConformalError, Coverage, Threshold,
};

/// A calibrated nonconformity threshold from [`calibrate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalThreshold {
    /// The calibration order statistic `q̂`, on the score scale supplied to
    /// [`calibrate_scores`]. It is `f32::INFINITY` when the calibration set is
    /// too small for the requested confidence (conservative fallback).
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
    /// A raw calibration score at `index` was NaN or infinite.
    NonFiniteScore {
        /// Position of the invalid score in the input slice.
        index: usize,
    },
    /// A calibration answer id is out of range for the scorer.
    AnswerOutOfRange,
}

impl std::fmt::Display for ConformalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha => write!(f, "alpha must be in (0, 1)"),
            Self::NoCalibrationExamples => write!(f, "calibration set is empty"),
            Self::NonFiniteScore { index } => {
                write!(f, "calibration score at index {index} must be finite")
            }
            Self::AnswerOutOfRange => write!(f, "calibration answer id out of range"),
        }
    }
}

impl std::error::Error for ConformalError {}

fn map_statskit_error(error: StatskitConformalError) -> ConformalError {
    match error {
        StatskitConformalError::InvalidCoverage => ConformalError::InvalidAlpha,
        StatskitConformalError::EmptyCalibration => ConformalError::NoCalibrationExamples,
        StatskitConformalError::NonFiniteScore { index } => {
            ConformalError::NonFiniteScore { index }
        }
    }
}

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
/// [`ConformalError::AnswerOutOfRange`] if an answer id is not an entity; or
/// [`ConformalError::NonFiniteScore`] if a scorer violates its degree contract.
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
/// finite exchangeable score works), return the finite-sample conformal
/// threshold: the `ceil((n + 1) * (1 - alpha))`-th smallest score, or
/// `INFINITY` when the rank exceeds `n` (conservative fallback).
/// The rank uses the caller's represented binary `f32` alpha exactly after
/// conversion to `f64`; alpha is not rounded to a decimal approximation.
/// Calibration and future scores must use the same fixed scoring rule and
/// be exchangeable; the score rule must not be fitted on calibration examples.
///
/// For a custom readout, supply one score per calibration example and form
/// prediction sets on the same score scale. Use [`answer_set_from_degrees`]
/// only when the score is `1 - degree`.
///
/// # Errors
///
/// [`ConformalError::InvalidAlpha`] unless `0 < alpha < 1`;
/// [`ConformalError::NoCalibrationExamples`] on an empty slice;
/// [`ConformalError::NonFiniteScore`] if a score is NaN or infinite.
pub fn calibrate_scores(
    nonconformities: &[f32],
    alpha: f32,
) -> Result<ConformalThreshold, ConformalError> {
    // Validate alpha before inspecting the scores.
    let coverage = Coverage::from_miscoverage(f64::from(alpha)).map_err(map_statskit_error)?;
    // Finite f32 scores widen exactly; validation retains the original indices.
    let mut scores: Vec<f64> = nonconformities.iter().copied().map(f64::from).collect();

    let n = scores.len();
    let qhat = match calibrate_in_place(&mut scores, coverage).map_err(map_statskit_error)? {
        // The calibrator selects a supplied score, so this conversion reverses
        // the exact f32-to-f64 conversion above.
        Threshold::Finite(selected) => {
            let qhat = selected as f32;
            debug_assert_eq!(f64::from(qhat), selected);
            qhat
        }
        Threshold::Unbounded => f32::INFINITY,
    };
    Ok(ConformalThreshold {
        qhat,
        alpha,
        n_calibration: n,
    })
}

/// The conformal answer set: every entity whose `1 - degree` score is at most
/// `q̂`, with its degree, best first.
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
/// The scorer-agnostic core of [`answer_set`]: every entity whose `1 - degree`
/// score is at most `q̂`, best first, ties broken by id. `degrees[i]` is entity
/// `i`'s degree. Compare in score space to retain ties that rounding a
/// `1 - q̂` degree cutoff could exclude.
/// The threshold must come from [`calibrate`] or from [`calibrate_scores`] on
/// `1 - degree` nonconformities over the same `[0, 1]` degree scale. For another
/// score, use [`calibrate_scores`] and construct the matching prediction set in
/// the caller.
pub fn answer_set_from_degrees(
    degrees: &[f32],
    threshold: &ConformalThreshold,
) -> Vec<(usize, f32)> {
    let mut set: Vec<(usize, f32)> = degrees
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, d)| 1.0 - *d <= threshold.qhat)
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
/// applies the same `1 - degree <= q̂` cutoff, but only over candidates the
/// caller supplied. Its threshold has the same `1 - degree` requirement as
/// [`answer_set_from_degrees`]. When `q̂` is infinite, the conservative fallback
/// is the full candidate pool, not every possible entity. If a candidate id
/// appears more than once, the highest supplied degree is retained.
///
/// A candidate pool can omit the designated true answer. The full-entity
/// split-conformal coverage guarantee does not automatically carry over:
/// omitted true answers reduce coverage, even when `q̂` is infinite.
pub fn answer_set_from_scored_pool(
    scored: &[(usize, f32)],
    threshold: &ConformalThreshold,
) -> Vec<(usize, f32)> {
    let mut best_by_id = std::collections::BTreeMap::new();
    for &(entity, degree) in scored {
        if 1.0 - degree <= threshold.qhat {
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
    use proptest::prelude::*;

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
    /// the four calibration pairs is 3/4. This checks thresholding on the
    /// calibration data; it is not a held-out coverage estimate.
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

    #[test]
    fn raw_rank_and_gap_scores_keep_their_caller_scale() {
        // `subsume`'s learned-ranker readout uses a zero-based target rank as
        // one nonconformity. Its matching set constructor takes every item up
        // to `floor(qhat)`, so replacing a selected rank above one with 1.0
        // changes the set.
        let rank_threshold = calibrate_scores(&[0.0, 1.0, 2.0, 3.0], 0.5).unwrap();
        assert_eq!(rank_threshold.qhat, 2.0);
        let selected_rank_count = rank_threshold.qhat.floor() as usize + 1;
        assert_eq!(selected_rank_count, 3);

        // Its score-gap readout likewise builds a set by comparing raw scores
        // against `best_score - qhat`; the chosen order statistic may exceed
        // one without being invalid.
        let gap_threshold = calibrate_scores(&[0.25, 0.75, 1.75, 3.25], 0.5).unwrap();
        assert_eq!(gap_threshold.qhat, 1.75);
        let best_score = 3.25;
        let selected: Vec<_> = [3.25, 2.0, 1.5, -1.0]
            .into_iter()
            .filter(|score| *score >= best_score - gap_threshold.qhat)
            .collect();
        assert_eq!(selected, vec![3.25, 2.0, 1.5]);
    }

    #[test]
    fn finite_negative_raw_scores_are_ordered_without_clamping() {
        let threshold = calibrate_scores(&[-4.0, -3.0, -2.0, -1.0], 0.5).unwrap();
        assert_eq!(threshold.qhat, -2.0);
    }

    #[test]
    fn raw_scores_reject_each_nonfinite_value_before_sorting() {
        for (scores, index) in [
            (vec![0.1, f32::NAN, 0.3], 1),
            (vec![f32::INFINITY], 0),
            (vec![f32::NEG_INFINITY], 0),
        ] {
            assert_eq!(
                calibrate_scores(&scores, 0.5).unwrap_err(),
                ConformalError::NonFiniteScore { index }
            );
        }
    }

    #[test]
    fn raw_score_validation_keeps_invalid_alpha_before_empty_and_nonfinite_inputs() {
        assert_eq!(
            calibrate_scores(&[], 0.0).unwrap_err(),
            ConformalError::InvalidAlpha
        );
        assert_eq!(
            calibrate_scores(&[f32::NAN], 1.0).unwrap_err(),
            ConformalError::InvalidAlpha
        );
        assert_eq!(
            calibrate_scores(&[], 0.5).unwrap_err(),
            ConformalError::NoCalibrationExamples
        );
        assert_eq!(
            calibrate_scores(&[0.0, f32::NAN], 0.5).unwrap_err(),
            ConformalError::NonFiniteScore { index: 1 }
        );
    }

    #[test]
    fn statskit_adapter_preserves_signed_tied_selected_scores() {
        let alpha = 0.5f32;
        let scores = [-4.0f32, -1.0, -1.0, 2.0];
        let ours = calibrate_scores(&scores, alpha).unwrap();

        assert_eq!(ours.qhat, -1.0);
        assert_eq!(ours.n_calibration, 4);
    }

    #[test]
    fn raw_score_unbounded_rank_still_uses_the_conservative_fallback() {
        let alpha = 0.1f32;
        let scores = [-4.0f32, 7.0];
        let threshold = calibrate_scores(&scores, alpha).unwrap();
        assert!(threshold.qhat.is_infinite());
    }

    #[test]
    fn binary_alpha_boundary_selects_the_exact_rank() {
        let scores = [0.0f32, 1.0, 2.0, 3.0];
        let boundary = 0.2f32;
        assert_eq!(calibrate_scores(&scores, boundary).unwrap().qhat, 3.0);

        let just_below_boundary = f32::from_bits(boundary.to_bits() - 1);
        assert!(calibrate_scores(&scores, just_below_boundary)
            .unwrap()
            .qhat
            .is_infinite());
    }

    proptest! {
        #[test]
        fn raw_score_leave_one_out_rank_coverage_is_at_least_ninety_percent(
            scores in prop::collection::vec(-10_000_i16..10_000_i16, 2..64)
        ) {
            let scores: Vec<f32> = scores.into_iter().map(f32::from).collect();
            let covered = (0..scores.len()).filter(|&held_out| {
                let calibration: Vec<f32> = scores.iter().enumerate()
                    .filter(|(index, _)| *index != held_out)
                    .map(|(_, &score)| score)
                    .collect();
                let threshold = calibrate_scores(&calibration, 0.1_f32).unwrap();
                scores[held_out] <= threshold.qhat
            }).count();

            // `0.1_f32` is the caller-visible representable level. The rank
            // implementation evaluates its formula in f64; this integer check
            // expresses the nominal 90% finite-sample guarantee without a
            // floating-point comparison. Ties can only add coverage.
            prop_assert!(covered * 10 >= scores.len() * 9);
        }
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

    #[test]
    fn answer_sets_include_scores_equal_to_the_calibrated_threshold() {
        let degree = 0.1_f32;
        let threshold = calibrate_scores(&[1.0 - degree; 9], 0.1).unwrap();
        // In f32, subtracting this score from one does not recover the degree.
        assert!(1.0 - threshold.qhat > degree);
        assert_eq!(
            answer_set_from_degrees(&[degree, 0.01], &threshold),
            vec![(0, degree)]
        );
        assert_eq!(
            answer_set_from_scored_pool(&[(37, degree), (9, 0.01)], &threshold),
            vec![(37, degree)]
        );
    }

    proptest! {
        #[test]
        fn degree_answer_sets_retain_calibrated_ties(degree in 0.0_f32..=1.0) {
            let threshold = calibrate_scores(&[1.0 - degree], 0.5).unwrap();
            prop_assert_eq!(answer_set_from_degrees(&[degree], &threshold), vec![(0, degree)]);
            prop_assert_eq!(answer_set_from_scored_pool(&[(37, degree)], &threshold), vec![(37, degree)]);
        }
    }

    /// answer_set_from_degrees applies the nonconformity cutoff to a raw degree
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
        let mut scored = [(10usize, 0.9f32), (5, 0.65), (7, 0.75), (10, 0.85)];
        let set = answer_set_from_scored_pool(&scored, &thr);
        assert_eq!(set, vec![(10, 0.9), (7, 0.75)]);
        scored.reverse();
        assert_eq!(answer_set_from_scored_pool(&scored, &thr), set);
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
