//! The tranz point-model adapter.
use crate::query::{AtomicScorer, RawProjection, RawScoreOrder};

/// Wraps a `tranz` point-embedding model (`TransE`/`RotatE`/`ComplEx`/
/// `DistMult`) as an [`AtomicScorer`].
///
/// `tranz` scores are distances or negative similarities where **lower means
/// more likely**; this maps them to `[0, 1]` membership degrees via
/// `sigmoid(-score / temperature)`. Trained 1-N models produce large-margin
/// scores that saturate a plain sigmoid (fine for ranking, degenerate for
/// calibrated degrees and conformal thresholds); a temperature above 1
/// spreads the degrees without changing any ranking (the map stays strictly
/// monotone).
///
/// ```no_run
/// use heyting::{answer_query_topk, Godel, Query, QueryConfig};
/// use heyting::adapters::PointModel;
///
/// # let (entity_vecs, relation_vecs, dim) =
/// #     (vec![vec![0.0_f32]], vec![vec![0.0_f32]], 1);
/// // any tranz::Scorer: a trained DistMult/ComplEx/TransE/RotatE.
/// let model = tranz::DistMult::from_vecs(entity_vecs, relation_vecs, dim);
/// let scorer = PointModel::new(model);
/// let q = Query::anchor(0, 0).then(1); // 2-hop chain
/// let top = answer_query_topk::<Godel>(&scorer, &q, &QueryConfig::default(), 10);
/// ```
pub struct PointModel<S> {
    /// The wrapped tranz scorer.
    pub model: S,
    /// Sigmoid temperature; `1.0` is the plain logistic map.
    pub temperature: f32,
}

impl<S> PointModel<S> {
    /// Wrap a scorer with the plain sigmoid map (temperature `1.0`).
    pub fn new(model: S) -> Self {
        Self::with_temperature(model, 1.0)
    }

    /// Wrap a scorer with a temperature; non-finite or non-positive values
    /// fall back to `1.0` (the map must stay strictly monotone).
    pub fn with_temperature(model: S, temperature: f32) -> Self {
        let temperature = if temperature.is_finite() && temperature > 0.0 {
            temperature
        } else {
            1.0
        };
        Self { model, temperature }
    }
}

impl<S: tranz::Scorer> AtomicScorer for PointModel<S> {
    fn num_entities(&self) -> usize {
        self.model.num_entities()
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        self.model
            .score_all_tails(anchor, relation)
            .iter()
            .map(|&s| sigmoid(-s / self.temperature))
            .collect()
    }

    fn project_batch(&self, anchors: &[usize], relation: usize) -> Vec<Vec<f32>> {
        self.model
            .score_all_tails_batch(anchors, relation)
            .into_iter()
            .map(|scores| {
                scores
                    .into_iter()
                    .map(|s| sigmoid(-s / self.temperature))
                    .collect()
            })
            .collect()
    }

    fn project_raw(&self, anchor: usize, relation: usize) -> Option<RawProjection> {
        Some(RawProjection::new(
            self.model.score_all_tails(anchor, relation),
            RawScoreOrder::LowerIsBetter,
        ))
    }

    fn project_raw_batch(&self, anchors: &[usize], relation: usize) -> Option<Vec<RawProjection>> {
        Some(
            self.model
                .score_all_tails_batch(anchors, relation)
                .into_iter()
                .map(|scores| RawProjection::new(scores, RawScoreOrder::LowerIsBetter))
                .collect(),
        )
    }

    fn project_subset(&self, anchor: usize, relation: usize, candidates: &[usize]) -> Vec<f32> {
        let n = self.model.num_entities();
        candidates
            .iter()
            .map(|&tail| {
                if tail < n {
                    sigmoid(-self.model.score(anchor, relation, tail) / self.temperature)
                } else {
                    0.0
                }
            })
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
        let scorer = PointModel::new(model);

        let scores = answer_query::<Godel>(&scorer, &Query::anchor(0, 0), &QueryConfig::default());
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

    #[test]
    fn point_model_subset_matches_dense_projection() {
        let ent = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let rel = vec![vec![1.0, 1.0]];
        let scorer = PointModel::new(tranz::DistMult::from_vecs(ent, rel, 2));

        let dense = scorer.project(0, 0);
        assert_eq!(
            scorer.project_subset(0, 0, &[2, 0, 99]),
            vec![dense[2], dense[0], 0.0]
        );
    }

    #[test]
    fn point_model_batch_projection_matches_dense_projection() {
        let ent = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let rel = vec![vec![1.0, 1.0]];
        let scorer = PointModel::new(tranz::DistMult::from_vecs(ent, rel, 2));

        assert_eq!(
            scorer.project_batch(&[0, 1], 0),
            vec![scorer.project(0, 0), scorer.project(1, 0),]
        );
    }

    #[test]
    fn point_model_exposes_lower_is_better_raw_scores() {
        let ent = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let rel = vec![vec![1.0, 1.0]];
        let scorer = PointModel::new(tranz::DistMult::from_vecs(ent, rel, 2));

        let raw = scorer.project_raw(0, 0).unwrap();
        assert_eq!(raw.order, RawScoreOrder::LowerIsBetter);
        assert_eq!(raw.len(), 3);
    }
}
