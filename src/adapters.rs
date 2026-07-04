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
        pub(super) fn query_box(
            &self,
            anchor: usize,
            relation: usize,
        ) -> Option<(Vec<f32>, &[f32])> {
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
mod box_dnf {
    //! The geometric execution mode: materialize a query's answer REGION.
    //!
    //! Where the engine composes degrees with t-norms, this executor composes
    //! the boxes themselves, Query2Box-style: hops translate and widen,
    //! conjunction is exact box intersection (axis-aligned boxes are closed
    //! under it), disjunction is a DNF union-of-boxes list (boxes are NOT
    //! closed under union), and negation is unsupported outright (box
    //! complements are not boxes; that gap is BetaE's founding motivation).
    //! The artifact is therefore a [`BoxDnf`], not a single box, and the
    //! composition tree is returned as an [`Explanation`] — the answer's
    //! proof sketch: which anchors, which translations, which intersections.
    //!
    //! The two modes answer differently by design: the t-norm path relaxes
    //! logic over any geometry; this path IS the geometry, exact for
    //! intersection but box-by-fiat for projection. Divergence between them
    //! measures what the t-norm relaxation costs on a given model.

    use super::subsume_adapter::BoxModel;
    use crate::query::Query;

    /// One axis-aligned query box (center + half-width offsets).
    #[derive(Debug, Clone, PartialEq)]
    pub struct QueryBox {
        /// Box center per dimension.
        pub center: Vec<f32>,
        /// Half-width per dimension (non-negative).
        pub offset: Vec<f32>,
    }

    impl QueryBox {
        /// Exact intersection; `None` when empty in some dimension.
        fn intersect(&self, other: &Self) -> Option<Self> {
            let d = self.center.len();
            let mut center = Vec::with_capacity(d);
            let mut offset = Vec::with_capacity(d);
            for i in 0..d {
                let lo = (self.center[i] - self.offset[i]).max(other.center[i] - other.offset[i]);
                let hi = (self.center[i] + self.offset[i]).min(other.center[i] + other.offset[i]);
                if lo > hi {
                    return None;
                }
                center.push((lo + hi) * 0.5);
                offset.push((hi - lo) * 0.5);
            }
            Some(Self { center, offset })
        }

        /// Natural-log volume; `-inf` for a degenerate (zero-width) box.
        pub fn log_volume(&self) -> f32 {
            self.offset.iter().map(|o| (2.0 * o).ln()).sum()
        }
    }

    /// A union of boxes: the materialized answer region of an EPFO query.
    #[derive(Debug, Clone, PartialEq)]
    pub struct BoxDnf {
        /// Disjuncts; empty means the answer region is provably empty.
        pub boxes: Vec<QueryBox>,
    }

    impl BoxDnf {
        /// Membership degree of a point: `exp(-d / temperature)` under the
        /// alpha-weighted Query2Box distance, maximized over disjuncts.
        /// Comparable with [`BoxModel`]'s atomic degrees by construction.
        pub fn degree(&self, point: &[f32], alpha: f32, temperature: f32) -> f32 {
            self.boxes
                .iter()
                .filter_map(|b| {
                    subsume::distance::query2box_distance(&b.center, &b.offset, point, alpha)
                        .ok()
                        .map(|d| (-d / temperature).exp())
                })
                .fold(0.0, f32::max)
        }

        /// Upper bound on the region's log-volume (log of summed disjunct
        /// volumes; exact when disjuncts are disjoint). The free cardinality
        /// estimate for planning.
        pub fn log_volume_bound(&self) -> f32 {
            let sum: f32 = self.boxes.iter().map(|b| b.log_volume().exp()).sum();
            sum.ln()
        }
    }

    /// Why a query cannot be materialized geometrically.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum MaterializeError {
        /// Negation, implication, and `Given` leaves have no box form.
        UnsupportedConnective(&'static str),
        /// An anchor's entity or relation id is out of range. Unlike degree
        /// mode's zero convention, an explicit plan fails loudly.
        UnknownId,
    }

    impl std::fmt::Display for MaterializeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::UnsupportedConnective(c) => {
                    write!(f, "{c} has no box materialization (boxes are closed under intersection only; disjunction is DNF)")
                }
                Self::UnknownId => write!(f, "anchor entity or relation id out of range"),
            }
        }
    }

    impl std::error::Error for MaterializeError {}

    /// The composition tree behind a materialized region: the answer's
    /// witness chain, one node per query connective.
    #[derive(Debug, Clone)]
    pub struct Explanation {
        /// Human-readable node label (`anchor(2, 0)`, `and`, `or`, `then(1)`).
        pub label: String,
        /// The region materialized at this node.
        pub region: BoxDnf,
        /// Sub-explanations, in query order.
        pub children: Vec<Explanation>,
    }

    impl Explanation {
        /// Indented one-line-per-node rendering with box counts and volumes.
        pub fn render(&self) -> String {
            let mut out = String::new();
            self.render_into(&mut out, 0);
            out
        }

        fn render_into(&self, out: &mut String, depth: usize) {
            use std::fmt::Write;
            let vol = self.region.log_volume_bound();
            let _ = writeln!(
                out,
                "{}{} [{} box(es), log-vol {:.2}]",
                "  ".repeat(depth),
                self.label,
                self.region.boxes.len(),
                vol
            );
            for c in &self.children {
                c.render_into(out, depth + 1);
            }
        }
    }

    impl BoxModel {
        /// Materialize the answer region of an EPFO query, with its
        /// composition tree. See the module doc for what is and is not
        /// expressible; [`materialize`](Self::materialize) drops the tree.
        pub fn materialize_explained(
            &self,
            query: &Query,
        ) -> Result<Explanation, MaterializeError> {
            match query {
                Query::Anchor { entity, relation } => {
                    let (center, offset) = self
                        .query_box(*entity, *relation)
                        .ok_or(MaterializeError::UnknownId)?;
                    Ok(Explanation {
                        label: format!("anchor({entity}, {relation})"),
                        region: BoxDnf {
                            boxes: vec![QueryBox {
                                center,
                                offset: offset.to_vec(),
                            }],
                        },
                        children: vec![],
                    })
                }
                Query::Project { inner, relation } => {
                    let child = self.materialize_explained(inner)?;
                    let (trans, widen) = self
                        .relation_parts(*relation)
                        .ok_or(MaterializeError::UnknownId)?;
                    let boxes = child
                        .region
                        .boxes
                        .iter()
                        .map(|b| QueryBox {
                            center: b.center.iter().zip(trans).map(|(c, t)| c + t).collect(),
                            offset: b.offset.iter().zip(widen).map(|(o, w)| o + w).collect(),
                        })
                        .collect();
                    Ok(Explanation {
                        label: format!("then({relation})"),
                        region: BoxDnf { boxes },
                        children: vec![child],
                    })
                }
                Query::Intersection { branches } => {
                    let children: Vec<Explanation> = branches
                        .iter()
                        .map(|b| self.materialize_explained(b))
                        .collect::<Result<_, _>>()?;
                    // Exact DNF intersection: distribute over disjuncts,
                    // drop empty combinations.
                    let mut acc: Vec<QueryBox> = match children.first() {
                        Some(c) => c.region.boxes.clone(),
                        None => vec![],
                    };
                    for c in children.iter().skip(1) {
                        acc = acc
                            .iter()
                            .flat_map(|a| c.region.boxes.iter().filter_map(|b| a.intersect(b)))
                            .collect();
                    }
                    Ok(Explanation {
                        label: "and".into(),
                        region: BoxDnf { boxes: acc },
                        children,
                    })
                }
                Query::Union { branches } => {
                    let children: Vec<Explanation> = branches
                        .iter()
                        .map(|b| self.materialize_explained(b))
                        .collect::<Result<_, _>>()?;
                    let boxes = children
                        .iter()
                        .flat_map(|c| c.region.boxes.iter().cloned())
                        .collect();
                    Ok(Explanation {
                        label: "or".into(),
                        region: BoxDnf { boxes },
                        children,
                    })
                }
                Query::Negation { .. } => Err(MaterializeError::UnsupportedConnective("negation")),
                Query::Implication { .. } => {
                    Err(MaterializeError::UnsupportedConnective("implication"))
                }
                Query::Given { .. } => Err(MaterializeError::UnsupportedConnective("a Given leaf")),
            }
        }

        /// The materialized answer region alone.
        pub fn materialize(&self, query: &Query) -> Result<BoxDnf, MaterializeError> {
            self.materialize_explained(query).map(|e| e.region)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn model() -> BoxModel {
            // 3 entities on a line; relation 0 translates +1 with box
            // half-width 0.25; relation 1 translates +2 with half-width 0.75.
            BoxModel::new(
                vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![4.0, 0.0]],
                vec![
                    (vec![1.0, 0.0], vec![0.25, 0.25]),
                    (vec![2.0, 0.0], vec![0.75, 0.75]),
                ],
                BoxModel::DEFAULT_ALPHA,
                1.0,
            )
            .unwrap()
        }

        #[test]
        fn anchor_materializes_the_translated_box() {
            let m = model();
            let r = m.materialize(&Query::anchor(0, 0)).unwrap();
            assert_eq!(r.boxes.len(), 1);
            assert_eq!(r.boxes[0].center, vec![1.0, 0.0]);
            assert_eq!(r.boxes[0].offset, vec![0.25, 0.25]);
        }

        #[test]
        fn chains_accumulate_translation_and_width() {
            let m = model();
            let r = m.materialize(&Query::anchor(0, 0).then(1)).unwrap();
            assert_eq!(r.boxes[0].center, vec![3.0, 0.0]);
            assert_eq!(r.boxes[0].offset, vec![1.0, 1.0]);
        }

        /// Hand-computed exact intersection: boxes [0.75, 1.25] and
        /// [1.0, 3.5] on axis 0 intersect to [1.0, 1.25].
        #[test]
        fn intersection_is_exact() {
            let m = model();
            let q = Query::intersection(vec![
                Query::anchor(0, 0), // center 1.0, offset 0.25 -> [0.75, 1.25]
                Query::anchor(0, 1), // center 2.0, offset 0.75 -> [1.25, 2.75]
            ]);
            let r = m.materialize(&q).unwrap();
            assert_eq!(r.boxes.len(), 1);
            assert!((r.boxes[0].center[0] - 1.25).abs() < 1e-6);
            assert!((r.boxes[0].offset[0] - 0.0).abs() < 1e-6);
        }

        /// Disjoint conjunction materializes to a provably-empty region.
        #[test]
        fn empty_intersection_is_expressible() {
            let m = model();
            let q = Query::intersection(vec![
                Query::anchor(0, 0), // [0.75, 1.25]
                Query::anchor(2, 0), // centered at 5.0: [4.75, 5.25]
            ]);
            let r = m.materialize(&q).unwrap();
            assert!(r.boxes.is_empty());
        }

        /// Union is DNF: the artifact carries both boxes, and degree takes
        /// the max over disjuncts.
        #[test]
        fn union_is_dnf() {
            let m = model();
            let q = Query::union(vec![Query::anchor(0, 0), Query::anchor(2, 0)]);
            let r = m.materialize(&q).unwrap();
            assert_eq!(r.boxes.len(), 2);
            assert!(r.degree(&[1.0, 0.0], 0.02, 1.0) > 0.9);
            assert!(r.degree(&[5.0, 0.0], 0.02, 1.0) > 0.9);
        }

        /// On a single-box query the materialized degree IS the atomic
        /// degree: the two execution modes coincide exactly at atoms.
        #[test]
        fn atomic_degrees_agree_across_modes() {
            use crate::query::AtomicScorer;
            let m = model();
            let q = Query::anchor(0, 0);
            let region = m.materialize(&q).unwrap();
            let dense = m.project(0, 0);
            for (e, point) in [(0, [0.0, 0.0]), (1, [1.0, 0.0]), (2, [4.0, 0.0])] {
                let g = region.degree(&point, BoxModel::DEFAULT_ALPHA, 1.0);
                assert!(
                    (g - dense[e]).abs() < 1e-6,
                    "entity {e}: {g} vs {}",
                    dense[e]
                );
            }
        }

        #[test]
        fn unsupported_connectives_fail_loudly() {
            let m = model();
            assert_eq!(
                m.materialize(&Query::anchor(0, 0).negate()).unwrap_err(),
                MaterializeError::UnsupportedConnective("negation")
            );
            assert_eq!(
                m.materialize(&Query::given(vec![1.0])).unwrap_err(),
                MaterializeError::UnsupportedConnective("a Given leaf")
            );
            assert_eq!(
                m.materialize(&Query::anchor(9, 0)).unwrap_err(),
                MaterializeError::UnknownId
            );
        }

        #[test]
        fn explanation_renders_the_witness_chain() {
            let m = model();
            let q = Query::intersection(vec![Query::anchor(0, 0), Query::anchor(0, 1)]);
            let e = m.materialize_explained(&q).unwrap();
            let text = e.render();
            assert!(text.contains("and"), "{text}");
            assert!(text.contains("anchor(0, 0)"), "{text}");
            assert!(text.contains("anchor(0, 1)"), "{text}");
        }
    }
}

#[cfg(feature = "subsume")]
pub use box_dnf::{BoxDnf, Explanation, MaterializeError, QueryBox};

#[cfg(feature = "subsume")]
pub use subsume_adapter::{BoxModel, BoxModelError};
