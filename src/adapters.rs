//! Adapters from ecosystem models to [`AtomicScorer`](crate::AtomicScorer).
//!
//! Each adapter is behind an optional feature so the core stays dependency-free.
//! They are the worked proof that the [`AtomicScorer`](crate::AtomicScorer) seam
//! carries real trained embeddings, not just the in-memory [`crate::FuzzyKg`].

#[cfg(feature = "tranz")]
mod tranz_adapter {
    use crate::query::AtomicScorer;

    /// Wraps a `tranz` point-embedding model (`TransE`/`RotatE`/`ComplEx`/
    /// `DistMult`) as an [`AtomicScorer`].
    ///
    /// `tranz` scores are distances or negative similarities where **lower means
    /// more likely**; this maps them to `[0, 1]` membership degrees via
    /// `sigmoid(-score)`, so the query engine sees calibrated degrees.
    ///
    /// ```no_run
    /// use heyting::{answer_query_topk, Godel, Query, QueryConfig};
    /// use heyting::adapters::PointModel;
    ///
    /// # let (entity_vecs, relation_vecs, dim) =
    /// #     (vec![vec![0.0_f32]], vec![vec![0.0_f32]], 1);
    /// // any tranz::Scorer: a trained DistMult/ComplEx/TransE/RotatE.
    /// let model = tranz::DistMult::from_vecs(entity_vecs, relation_vecs, dim);
    /// let scorer = PointModel(model);
    /// let q = Query::anchor(0, 0).then(1); // 2-hop chain
    /// let top = answer_query_topk::<Godel>(&scorer, &q, &QueryConfig::default(), 10);
    /// ```
    pub struct PointModel<S>(pub S);

    impl<S: tranz::Scorer> AtomicScorer for PointModel<S> {
        fn num_entities(&self) -> usize {
            self.0.num_entities()
        }

        fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
            self.0
                .score_all_tails(anchor, relation)
                .iter()
                .map(|&s| sigmoid(-s))
                .collect()
        }
    }

    /// Numerically stable logistic sigmoid.
    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{answer_query, Godel, Query, QueryConfig};

        #[test]
        fn point_model_yields_valid_membership_degrees() {
            // 3 entities, 1 relation, dim 2. DistMult scores via the trilinear
            // product; exact values do not matter, only that the adapter feeds
            // the engine calibrated [0, 1] degrees and a query runs end to end.
            let ent = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
            let rel = vec![vec![1.0, 1.0]];
            let model = tranz::DistMult::from_vecs(ent, rel, 2);
            let scorer = PointModel(model);

            let scores =
                answer_query::<Godel>(&scorer, &Query::anchor(0, 0), &QueryConfig::default());
            assert_eq!(scores.len(), 3);
            assert!(
                scores.iter().all(|&s| (0.0..=1.0).contains(&s)),
                "adapter must produce [0,1] degrees, got {scores:?}"
            );

            // A negated query under the same algebra stays in range too.
            let neg = answer_query::<Godel>(
                &scorer,
                &Query::anchor(0, 0).negate(),
                &QueryConfig::default(),
            );
            assert!(neg.iter().all(|&s| (0.0..=1.0).contains(&s)));
        }
    }
}

#[cfg(feature = "tranz")]
pub use tranz_adapter::PointModel;

#[cfg(feature = "subsume")]
mod subsume_adapter {
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
        fn query_box(&self, anchor: usize, relation: usize) -> Option<(Vec<f32>, &[f32])> {
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

            let scores =
                answer_query::<Godel>(&model, &Query::anchor(0, 0), &QueryConfig::default());
            assert_eq!(scores.len(), 3);
            assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
            assert!(
                scores[1] > 0.9,
                "in-box entity should be near 1: {scores:?}"
            );
            assert!(scores[1] > scores[0] && scores[0] > scores[2], "{scores:?}");

            let top = answer_query_topk::<Godel>(
                &model,
                &Query::anchor(0, 0),
                &QueryConfig::default(),
                1,
            );
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
}

#[cfg(feature = "subsume")]
pub use subsume_adapter::{BoxModel, BoxModelError};
