//! The subsume Query2Box-style box scorer.
use crate::query::AtomicScorer;

/// Query2Box-style scorer over trained box embeddings.
///
/// Entities are points; a relation is a `(translation, offset)` pair.
/// `project(anchor, r)` forms the query box `(point[anchor] +
/// translation[r], offset[r])` and scores every entity point by
/// `subsume`'s alpha-weighted Query2Box distance, mapped to a `[0, 1]`
/// degree via `exp(-distance / temperature)`: `1` at the box center,
/// decaying with distance outside.
///
/// This is the box counterpart of [`PointModel`](super::PointModel):
/// where that maps arbitrary link-prediction scores through a sigmoid,
/// boxes give a geometric membership degree. Chains, intersections, and
/// unions still compose in the engine's [`crate::Truth`] algebra
/// (level 1 of the retrieval-seam design); this adapter does not
/// materialize composed boxes.
#[derive(Debug, Clone)]
pub struct BoxModel {
    entity_points: Vec<Vec<f32>>,
    /// Per relation: (center translation, query-box offset).
    relations: Vec<(Vec<f32>, Vec<f32>)>,
    alpha: f32,
    temperature: f32,
}

/// Construction problems [`BoxModel::new`] rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoxModelError {
    /// An entity point or relation vector has the wrong dimension.
    DimensionMismatch,
    /// `alpha` is non-finite or `temperature` is not positive.
    InvalidParameter,
}

impl std::fmt::Display for BoxModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch => {
                write!(f, "entity and relation vectors must share one dimension")
            }
            Self::InvalidParameter => {
                write!(f, "alpha must be finite and temperature positive")
            }
        }
    }
}

impl std::error::Error for BoxModelError {}

impl BoxModel {
    /// Query2Box's usual inside-distance weight.
    pub const DEFAULT_ALPHA: f32 = 0.02;

    /// Build a scorer from trained entity points and per-relation
    /// `(translation, offset)` pairs. `alpha` weights the inside
    /// distance (Query2Box uses `0.02`); `temperature` scales the
    /// distance-to-degree map `exp(-d / temperature)`.
    pub fn new(
        entity_points: Vec<Vec<f32>>,
        relations: Vec<(Vec<f32>, Vec<f32>)>,
        alpha: f32,
        temperature: f32,
    ) -> Result<Self, BoxModelError> {
        if !alpha.is_finite() || !temperature.is_finite() || temperature <= 0.0 {
            return Err(BoxModelError::InvalidParameter);
        }
        let dim = entity_points.first().map(Vec::len).unwrap_or(0);
        if entity_points.iter().any(|p| p.len() != dim)
            || relations
                .iter()
                .any(|(t, o)| t.len() != dim || o.len() != dim)
        {
            return Err(BoxModelError::DimensionMismatch);
        }
        Ok(Self {
            entity_points,
            relations,
            alpha,
            temperature,
        })
    }

    /// Degree of `entity` under the query box `(query_center, offset)`.
    fn degree(&self, query_center: &[f32], offset: &[f32], entity: usize) -> f32 {
        let point = &self.entity_points[entity];
        match subsume::distance::query2box_distance(query_center, offset, point, self.alpha) {
            Ok(d) => (-d / self.temperature).exp(),
            // Dimensions are validated at construction, so this arm is
            // unreachable; degree 0 is the engine's "not an answer"
            // convention and keeps the scoring loop panic-free.
            Err(_) => 0.0,
        }
    }

    /// The query box for `(anchor, relation)`, if both ids are in range.
    pub(super) fn query_box(&self, anchor: usize, relation: usize) -> Option<(Vec<f32>, &[f32])> {
        let point = self.entity_points.get(anchor)?;
        let (translation, offset) = self.relations.get(relation)?;
        let center = point
            .iter()
            .zip(translation.iter())
            .map(|(p, t)| p + t)
            .collect();
        Some((center, offset.as_slice()))
    }
}

impl BoxModel {
    /// A relation's `(translation, widening)` pair, if in range.
    pub(super) fn relation_parts(&self, relation: usize) -> Option<(&[f32], &[f32])> {
        self.relations
            .get(relation)
            .map(|(t, o)| (t.as_slice(), o.as_slice()))
    }
}

impl AtomicScorer for BoxModel {
    fn num_entities(&self) -> usize {
        self.entity_points.len()
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        let n = self.num_entities();
        let Some((center, offset)) = self.query_box(anchor, relation) else {
            return vec![0.0; n];
        };
        (0..n).map(|e| self.degree(&center, offset, e)).collect()
    }

    fn project_subset(&self, anchor: usize, relation: usize, candidates: &[usize]) -> Vec<f32> {
        let n = self.num_entities();
        let Some((center, offset)) = self.query_box(anchor, relation) else {
            return vec![0.0; candidates.len()];
        };
        candidates
            .iter()
            .map(|&e| {
                if e < n {
                    self.degree(&center, offset, e)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{answer_query, answer_query_topk, Godel, Query, QueryConfig};

    /// 3 entities on a line; relation 0 translates by +1 with a tight box.
    /// From entity 0 the query box centers on entity 1: it must rank
    /// first with degree near 1, and degrees must be valid memberships.
    #[test]
    fn box_model_ranks_by_containment() {
        let entities = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![4.0, 0.0]];
        let relations = vec![(vec![1.0, 0.0], vec![0.25, 0.25])];
        let model = BoxModel::new(entities, relations, BoxModel::DEFAULT_ALPHA, 1.0).unwrap();

        let scores = answer_query::<Godel>(&model, &Query::anchor(0, 0), &QueryConfig::default());
        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
        assert!(
            scores[1] > 0.9,
            "in-box entity should be near 1: {scores:?}"
        );
        assert!(scores[1] > scores[0] && scores[0] > scores[2], "{scores:?}");

        let top =
            answer_query_topk::<Godel>(&model, &Query::anchor(0, 0), &QueryConfig::default(), 1);
        assert_eq!(top[0].0, 1);
    }

    /// project_subset scores only the requested candidates, aligned.
    #[test]
    fn subset_scoring_matches_dense() {
        use crate::query::AtomicScorer;
        let entities = vec![vec![0.0], vec![1.0], vec![2.0]];
        let relations = vec![(vec![1.0], vec![0.5])];
        let model = BoxModel::new(entities, relations, BoxModel::DEFAULT_ALPHA, 1.0).unwrap();
        let dense = model.project(0, 0);
        let subset = model.project_subset(0, 0, &[2, 0]);
        assert!((subset[0] - dense[2]).abs() < 1e-6);
        assert!((subset[1] - dense[0]).abs() < 1e-6);
    }

    #[test]
    fn rejects_mismatched_dimensions_and_bad_params() {
        assert_eq!(
            BoxModel::new(
                vec![vec![0.0, 0.0], vec![1.0]],
                vec![],
                BoxModel::DEFAULT_ALPHA,
                1.0
            )
            .unwrap_err(),
            BoxModelError::DimensionMismatch
        );
        assert_eq!(
            BoxModel::new(
                vec![vec![0.0]],
                vec![(vec![0.0, 0.0], vec![0.0])],
                0.02,
                1.0
            )
            .unwrap_err(),
            BoxModelError::DimensionMismatch
        );
        assert_eq!(
            BoxModel::new(vec![vec![0.0]], vec![], 0.02, 0.0).unwrap_err(),
            BoxModelError::InvalidParameter
        );
    }

    /// Out-of-range anchor or relation yields all-zero degrees (the
    /// engine's "not an answer" convention), never a panic.
    #[test]
    fn out_of_range_ids_score_zero() {
        let model = BoxModel::new(
            vec![vec![0.0]],
            vec![(vec![0.0], vec![1.0])],
            BoxModel::DEFAULT_ALPHA,
            1.0,
        )
        .unwrap();
        assert_eq!(model.project(5, 0), vec![0.0]);
        assert_eq!(model.project(0, 9), vec![0.0]);
    }
}
