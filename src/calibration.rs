//! Calibration wrappers for raw one-hop model scores.
//!
//! [`AtomicScorer`] query evaluation consumes membership
//! degrees in `[0, 1]`. This module keeps calibration explicit when a backend
//! can also expose native raw scores through
//! [`AtomicScorer::project_raw`].

use crate::query::{AtomicScorer, RawProjection};

/// Maps raw one-hop scores to membership degrees in `[0, 1]`.
pub trait Calibrator {
    /// Calibrate a raw projection for `relation`.
    fn calibrate(&self, relation: usize, raw: &RawProjection) -> Vec<f32>;
}

/// Wrap an [`AtomicScorer`] and replace its degree map with a calibrator when
/// raw scores are available.
#[derive(Debug, Clone)]
pub struct CalibratedScorer<S, C> {
    scorer: S,
    calibrator: C,
}

impl<S, C> CalibratedScorer<S, C> {
    /// Create a calibrated scorer wrapper.
    pub fn new(scorer: S, calibrator: C) -> Self {
        Self { scorer, calibrator }
    }

    /// The wrapped scorer.
    pub fn scorer(&self) -> &S {
        &self.scorer
    }

    /// The wrapped scorer, mutably.
    pub fn scorer_mut(&mut self) -> &mut S {
        &mut self.scorer
    }

    /// The calibrator.
    pub fn calibrator(&self) -> &C {
        &self.calibrator
    }
}

impl<S, C> AtomicScorer for CalibratedScorer<S, C>
where
    S: AtomicScorer,
    C: Calibrator,
{
    fn num_entities(&self) -> usize {
        self.scorer.num_entities()
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        if let Some(raw) = self.scorer.project_raw(anchor, relation) {
            self.calibrator.calibrate(relation, &raw)
        } else {
            self.scorer.project(anchor, relation)
        }
    }

    fn project_batch(&self, anchors: &[usize], relation: usize) -> Vec<Vec<f32>> {
        if let Some(raw_batch) = self.scorer.project_raw_batch(anchors, relation) {
            raw_batch
                .iter()
                .map(|raw| self.calibrator.calibrate(relation, raw))
                .collect()
        } else {
            self.scorer.project_batch(anchors, relation)
        }
    }

    fn project_raw(&self, anchor: usize, relation: usize) -> Option<RawProjection> {
        self.scorer.project_raw(anchor, relation)
    }

    fn project_raw_batch(&self, anchors: &[usize], relation: usize) -> Option<Vec<RawProjection>> {
        self.scorer.project_raw_batch(anchors, relation)
    }
}

/// Affine-logistic calibration over oriented raw scores.
///
/// If a raw score's order is lower-is-better, it is negated before applying
/// `sigmoid(scale * score + bias)`. Non-finite `scale` values and negative
/// scales fall back to `1.0`; non-finite bias falls back to `0.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineSigmoidCalibrator {
    scale: f32,
    bias: f32,
}

impl AffineSigmoidCalibrator {
    /// Create an affine-logistic calibrator.
    pub fn new(scale: f32, bias: f32) -> Self {
        let scale = if scale.is_finite() && scale >= 0.0 {
            scale
        } else {
            1.0
        };
        let bias = if bias.is_finite() { bias } else { 0.0 };
        Self { scale, bias }
    }

    /// The identity logistic map `sigmoid(score)`.
    pub const fn identity() -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
        }
    }

    /// Multiplicative scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Additive bias.
    pub fn bias(&self) -> f32 {
        self.bias
    }
}

impl Calibrator for AffineSigmoidCalibrator {
    fn calibrate(&self, _relation: usize, raw: &RawProjection) -> Vec<f32> {
        raw.scores
            .iter()
            .map(|&score| sigmoid(self.scale * raw.oriented_score(score) + self.bias))
            .collect()
    }
}

/// Row-softmax calibration over oriented raw scores.
///
/// Degrees are `softmax(score) * multiplier`, clipped to `max_degree`.
/// This matches the row-normalization shape used by QTO-style neural
/// adjacency matrices while keeping the output in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftmaxCalibrator {
    multiplier: f32,
    max_degree: f32,
}

impl SoftmaxCalibrator {
    /// Create a softmax calibrator.
    pub fn new(multiplier: f32, max_degree: f32) -> Self {
        let multiplier = if multiplier.is_finite() && multiplier > 0.0 {
            multiplier
        } else {
            1.0
        };
        let max_degree = if max_degree.is_finite() && max_degree > 0.0 {
            max_degree.min(1.0)
        } else {
            1.0
        };
        Self {
            multiplier,
            max_degree,
        }
    }

    /// A plain probability-row softmax.
    pub const fn probability_row() -> Self {
        Self {
            multiplier: 1.0,
            max_degree: 1.0,
        }
    }

    /// Multiplicative factor applied after softmax.
    pub fn multiplier(&self) -> f32 {
        self.multiplier
    }

    /// Maximum returned degree.
    pub fn max_degree(&self) -> f32 {
        self.max_degree
    }
}

impl Calibrator for SoftmaxCalibrator {
    fn calibrate(&self, _relation: usize, raw: &RawProjection) -> Vec<f32> {
        if raw.is_empty() {
            return Vec::new();
        }

        let oriented = raw.oriented_scores();
        let Some(max) = oriented
            .iter()
            .copied()
            .filter(|s| s.is_finite())
            .reduce(f32::max)
        else {
            return vec![0.0; raw.len()];
        };

        let exps: Vec<f32> = oriented
            .iter()
            .map(|&score| {
                if score.is_finite() {
                    (score - max).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 {
            return vec![0.0; raw.len()];
        }

        exps.iter()
            .map(|&e| ((e / sum) * self.multiplier).clamp(0.0, self.max_degree))
            .collect()
    }
}

fn sigmoid(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::RawScoreOrder;

    struct RawScorer;

    impl AtomicScorer for RawScorer {
        fn num_entities(&self) -> usize {
            3
        }

        fn project(&self, _anchor: usize, _relation: usize) -> Vec<f32> {
            vec![0.0; 3]
        }

        fn project_raw(&self, _anchor: usize, _relation: usize) -> Option<RawProjection> {
            Some(RawProjection::new(
                vec![0.0, 2.0, -2.0],
                RawScoreOrder::LowerIsBetter,
            ))
        }
    }

    #[test]
    fn raw_projection_orients_lower_is_better_scores() {
        let raw = RawProjection::new(vec![1.0, -2.0], RawScoreOrder::LowerIsBetter);
        assert_eq!(raw.oriented_scores(), vec![-1.0, 2.0]);
    }

    #[test]
    fn affine_sigmoid_preserves_oriented_order() {
        let raw = RawProjection::new(vec![0.0, 2.0, -2.0], RawScoreOrder::LowerIsBetter);
        let degrees = AffineSigmoidCalibrator::identity().calibrate(0, &raw);
        assert!(degrees[2] > degrees[0]);
        assert!(degrees[0] > degrees[1]);
    }

    #[test]
    fn calibrated_scorer_uses_raw_projection_when_available() {
        let scorer = CalibratedScorer::new(RawScorer, AffineSigmoidCalibrator::identity());
        let degrees = scorer.project(0, 0);
        assert!(degrees[2] > degrees[0]);
        assert!(degrees[0] > degrees[1]);
    }

    #[test]
    fn softmax_calibrator_returns_probability_row() {
        let raw = RawProjection::new(vec![2.0, 1.0, 0.0], RawScoreOrder::HigherIsBetter);
        let degrees = SoftmaxCalibrator::probability_row().calibrate(0, &raw);
        let sum: f32 = degrees.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(degrees[0] > degrees[1]);
        assert!(degrees[1] > degrees[2]);
    }
}
