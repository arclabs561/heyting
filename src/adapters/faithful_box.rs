//! Faithful EL++ box scorer: entities ARE boxes, subsumption is containment.
use crate::query::AtomicScorer;

/// Atomic scorer over faithful EL++ concept boxes.
///
/// The counterpart to the Query2Box `BoxModel` (feature `subsume`): there
/// entities are points and each relation is a `(translation, offset)` pair; here entities
/// themselves are boxes and the sole relation is graded subsumption `C ⊑ ?`,
/// the faithful EL++ geometry where box containment *is* the subsumption order.
///
/// `project(a, sub)` scores every concept `t` by the graded inclusion degree
/// `exp(-‖relu(|c_a − c_t| + o_a − o_t)‖ / temperature)`: `1` when box `a` sits
/// inside box `t` (so `a ⊑ t`), decaying as `a` protrudes from `t`. Any relation
/// other than the configured subsumption id projects all-zero, the engine's
/// "not an answer" convention rather than an error.
///
/// Unlike the other adapters this one pulls no ecosystem crate: the faithful
/// inclusion degree is a self-contained box formula, so it is always available.
/// Feed it the `centers` / `offsets` a `subsume` box trainer produces (one
/// `Vec<f32>` of length `dim` per concept) to run the query engine and
/// [`crate::conformal`] over an ontology-respecting geometric model.
#[derive(Debug, Clone)]
pub struct FaithfulBoxModel {
    centers: Vec<Vec<f32>>,
    offsets: Vec<Vec<f32>>,
    sub_relation: usize,
    temperature: f32,
}

/// Construction problems [`FaithfulBoxModel::new`] rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaithfulBoxError {
    /// `centers` and `offsets` differ in length, or an entity's center and
    /// offset differ in dimension.
    DimensionMismatch,
    /// `temperature` is not finite and positive.
    InvalidTemperature,
}

impl std::fmt::Display for FaithfulBoxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch => {
                write!(
                    f,
                    "centers and offsets must align and share one dimension per entity"
                )
            }
            Self::InvalidTemperature => write!(f, "temperature must be finite and positive"),
        }
    }
}

impl std::error::Error for FaithfulBoxError {}

impl FaithfulBoxModel {
    /// The reserved relation id for graded subsumption `C ⊑ ?` that
    /// [`FaithfulBoxModel::new`] configures by default via [`with_subsumption`].
    ///
    /// [`with_subsumption`]: FaithfulBoxModel::with_subsumption
    pub const DEFAULT_SUB: usize = 0;

    /// Build a scorer from per-concept box `centers` and `offsets` (one
    /// `Vec<f32>` of the shared dimension each), with subsumption on relation
    /// [`DEFAULT_SUB`](Self::DEFAULT_SUB) and temperature `1.0`.
    ///
    /// # Errors
    ///
    /// [`FaithfulBoxError::DimensionMismatch`] if the two outer lengths differ
    /// or any concept's center and offset differ in length.
    pub fn new(centers: Vec<Vec<f32>>, offsets: Vec<Vec<f32>>) -> Result<Self, FaithfulBoxError> {
        if centers.len() != offsets.len()
            || centers
                .iter()
                .zip(offsets.iter())
                .any(|(c, o)| c.len() != o.len())
        {
            return Err(FaithfulBoxError::DimensionMismatch);
        }
        Ok(Self {
            centers,
            offsets,
            sub_relation: Self::DEFAULT_SUB,
            temperature: 1.0,
        })
    }

    /// Set which relation id means graded subsumption (default
    /// [`DEFAULT_SUB`](Self::DEFAULT_SUB)).
    pub fn with_subsumption(mut self, relation: usize) -> Self {
        self.sub_relation = relation;
        self
    }

    /// Set the softness of the inclusion-degree map `exp(-d / temperature)`
    /// (default `1.0`). Smaller sharpens toward a hard containment indicator.
    ///
    /// # Errors
    ///
    /// [`FaithfulBoxError::InvalidTemperature`] unless `temperature` is finite
    /// and positive.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self, FaithfulBoxError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(FaithfulBoxError::InvalidTemperature);
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Graded inclusion degree of box `a` inside box `t` (`a ⊑ t`) in `[0, 1]`.
    fn degree(&self, a: usize, t: usize) -> f32 {
        let (ca, oa, ct, ot) = (
            &self.centers[a],
            &self.offsets[a],
            &self.centers[t],
            &self.offsets[t],
        );
        let mut acc = 0.0f32;
        for i in 0..ca.len() {
            // How far box a protrudes from box t along axis i; 0 when contained.
            let v = ((ca[i] - ct[i]).abs() + oa[i] - ot[i]).max(0.0);
            acc += v * v;
        }
        (-acc.sqrt() / self.temperature).exp()
    }
}

impl AtomicScorer for FaithfulBoxModel {
    fn num_entities(&self) -> usize {
        self.centers.len()
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        let n = self.num_entities();
        if relation != self.sub_relation || anchor >= n {
            return vec![0.0; n];
        }
        (0..n).map(|t| self.degree(anchor, t)).collect()
    }

    fn project_subset(&self, anchor: usize, relation: usize, candidates: &[usize]) -> Vec<f32> {
        let n = self.num_entities();
        if relation != self.sub_relation || anchor >= n {
            return vec![0.0; candidates.len()];
        }
        candidates
            .iter()
            .map(|&t| if t < n { self.degree(anchor, t) } else { 0.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{answer_query, answer_query_topk, Godel, Query, QueryConfig};

    // Thing(0) ⊒ mid(1) ⊒ leaf(2), 1-D nested boxes: leaf ⊂ mid ⊂ Thing.
    //        center  offset
    // 0 Thing (0.0)   (6.0)
    // 1 mid   (2.0)   (2.0)   inside Thing
    // 2 leaf  (2.5)   (0.5)   inside mid
    fn nested() -> FaithfulBoxModel {
        let centers = vec![vec![0.0], vec![2.0], vec![2.5]];
        let offsets = vec![vec![6.0], vec![2.0], vec![0.5]];
        FaithfulBoxModel::new(centers, offsets).unwrap()
    }

    #[test]
    fn subsumption_degrees_rank_true_ancestors_first() {
        let model = nested();
        // Superclasses of leaf(2): mid(1) and Thing(0) are true, both contain
        // leaf so both degrees are ~1; the tighter ancestor (mid) is at least as
        // strong. answer_query returns a degree per entity.
        let scores = answer_query::<Godel>(&model, &Query::anchor(2, 0), &QueryConfig::default());
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
        assert!(
            scores[1] > 0.9 && scores[0] > 0.9,
            "ancestors contain leaf: {scores:?}"
        );
        // Thing is NOT a subclass of leaf: degree(Thing ⊑ leaf) must be tiny.
        let up = answer_query::<Godel>(&model, &Query::anchor(0, 0), &QueryConfig::default());
        assert!(up[2] < 0.2, "root does not sit inside a leaf: {up:?}");
    }

    #[test]
    fn top_superclass_of_a_leaf_is_a_true_ancestor() {
        let model = nested();
        let top =
            answer_query_topk::<Godel>(&model, &Query::anchor(2, 0), &QueryConfig::default(), 2);
        let ids: Vec<usize> = top.iter().map(|&(e, _)| e).collect();
        assert!(
            ids.contains(&1) && ids.contains(&0),
            "top-2 are the ancestors: {ids:?}"
        );
    }

    #[test]
    fn non_subsumption_relation_projects_zero() {
        let model = nested();
        assert_eq!(model.project(2, 7), vec![0.0; 3]);
        assert_eq!(model.project(9, 0), vec![0.0; 3]); // out-of-range anchor
    }

    #[test]
    fn subset_scoring_matches_dense() {
        let model = nested();
        let dense = model.project(2, 0);
        let subset = model.project_subset(2, 0, &[1, 0]);
        assert!((subset[0] - dense[1]).abs() < 1e-6);
        assert!((subset[1] - dense[0]).abs() < 1e-6);
    }

    #[test]
    fn rejects_mismatched_dimensions_and_bad_temperature() {
        assert_eq!(
            FaithfulBoxModel::new(vec![vec![0.0, 0.0]], vec![vec![1.0]]).unwrap_err(),
            FaithfulBoxError::DimensionMismatch
        );
        assert_eq!(
            FaithfulBoxModel::new(vec![vec![0.0]], vec![vec![1.0]])
                .unwrap()
                .with_temperature(0.0)
                .unwrap_err(),
            FaithfulBoxError::InvalidTemperature
        );
    }
}
