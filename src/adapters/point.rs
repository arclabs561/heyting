//! The tranz point-model adapter.
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
}
