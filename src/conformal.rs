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
    let mut scores = Vec::with_capacity(examples.len());
    for (query, answer) in examples {
        if *answer >= n_entities {
            return Err(ConformalError::AnswerOutOfRange);
        }
        let degrees = answer_query::<T>(scorer, query, config);
        scores.push((1.0 - degrees[*answer]).clamp(0.0, 1.0));
    }
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
    let cutoff = 1.0 - threshold.qhat; // -inf when qhat is inf: everything.
    let mut set: Vec<(usize, f32)> = answer_query::<T>(scorer, query, config)
        .into_iter()
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
}
