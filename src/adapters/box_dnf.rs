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

use super::box_model::BoxModel;
use crate::eval::QueryAnswerReport;
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
        let max_log_volume = self
            .boxes
            .iter()
            .map(QueryBox::log_volume)
            .fold(f32::NEG_INFINITY, f32::max);
        if max_log_volume == f32::NEG_INFINITY {
            return f32::NEG_INFINITY;
        }

        let scaled_sum: f32 = self
            .boxes
            .iter()
            .map(|b| (b.log_volume() - max_log_volume).exp())
            .sum();
        max_log_volume + scaled_sum.ln()
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
    pub fn materialize_explained(&self, query: &Query) -> Result<Explanation, MaterializeError> {
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

    /// Materialize `query`, score every entity against the materialized region,
    /// and return top answers plus the region-volume cardinality estimate.
    pub fn materialized_answer_report(
        &self,
        query: &Query,
        k: usize,
    ) -> Result<QueryAnswerReport, MaterializeError> {
        let region = self.materialize(query)?;
        let (alpha, temperature) = self.scoring_params();
        let degrees: Vec<f32> = self
            .entity_points()
            .iter()
            .map(|point| region.degree(point, alpha, temperature))
            .collect();
        let mut top_k: Vec<(usize, f32)> = degrees.iter().copied().enumerate().collect();
        top_k.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        top_k.truncate(k);

        let log_volume = region.log_volume_bound();
        let predicted_cardinality = if log_volume.is_finite() {
            Some(log_volume.exp())
        } else {
            Some(0.0)
        };
        Ok(QueryAnswerReport {
            top_k,
            degrees,
            predicted_cardinality,
        })
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

    #[test]
    fn log_volume_bound_uses_logsumexp() {
        let b = QueryBox {
            center: vec![0.0; 200],
            offset: vec![1.0; 200],
        };
        let expected = b.log_volume() + 2.0_f32.ln();
        let dnf = BoxDnf { boxes: vec![b; 2] };
        let actual = dnf.log_volume_bound();
        assert!(actual.is_finite());
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }

    #[test]
    fn log_volume_bound_handles_empty_and_degenerate_regions() {
        assert_eq!(
            (BoxDnf { boxes: vec![] }).log_volume_bound(),
            f32::NEG_INFINITY
        );

        let degenerate = BoxDnf {
            boxes: vec![QueryBox {
                center: vec![0.0, 0.0],
                offset: vec![0.0, 1.0],
            }],
        };
        assert_eq!(degenerate.log_volume_bound(), f32::NEG_INFINITY);
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

    #[test]
    fn materialized_answer_report_includes_topk_and_cardinality_bound() {
        let m = model();
        let report = m
            .materialized_answer_report(&Query::anchor(0, 0), 2)
            .unwrap();
        assert_eq!(report.degrees.len(), 3);
        assert_eq!(report.top_k.len(), 2);
        assert_eq!(report.top_k[0].0, 1);
        assert!(report.predicted_cardinality.unwrap() > 0.0);
    }
}
