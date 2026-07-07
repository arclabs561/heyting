//! The tranz temporal-model adapter: trained timestamp-set hops.
//!
//! Wraps a `tranz::temporal::TemporalScorer` (a trained TComplEx) as an
//! [`AtomicScorer`], with the same virtual-relation registration pattern as
//! [`TemporalKg`](crate::TemporalKg): register `(base relation, TimeSet)`
//! and hop through the returned id anywhere a relation id goes. A hop's
//! degree is existential over the set — `max over τ ∈ S` of the calibrated
//! atom degree — which is the trained counterpart of `TemporalKg` taking
//! the best admitted fact.

use crate::query::AtomicScorer;
use crate::temporal::TimeSet;

/// Wraps a trained temporal scorer as an [`AtomicScorer`] with
/// timestamp-set virtual relations.
///
/// Base relation ids `0..num_relations` hop with no temporal constraint
/// (the whole axis); [`windowed`](Self::windowed) registers time-scoped
/// views at fresh ids. Energies map to degrees via
/// `sigmoid(-score / temperature)`, exactly as
/// [`PointModel`](crate::adapters::PointModel).
pub struct TemporalPointModel<S> {
    /// The wrapped tranz temporal scorer.
    pub model: S,
    /// Sigmoid temperature; `1.0` is the plain logistic map.
    pub temperature: f32,
    virtuals: Vec<(usize, TimeSet)>,
}

impl<S: tranz::temporal::TemporalScorer> TemporalPointModel<S> {
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
        Self {
            model,
            temperature,
            virtuals: Vec::new(),
        }
    }

    /// Register a time-scoped view of `relation` over the timestamp set
    /// and return its virtual relation id. `None` if `relation` is not a
    /// base relation or the set's axis does not match the model's.
    pub fn windowed(&mut self, relation: usize, times: TimeSet) -> Option<usize> {
        if relation >= self.model.num_relations()
            || times.num_timestamps() != self.model.num_timestamps()
        {
            return None;
        }
        self.virtuals.push((relation, times));
        Some(self.model.num_relations() + self.virtuals.len() - 1)
    }

    /// Resolve a (possibly virtual) relation id to `(base, TimeSet)`.
    fn resolve(&self, relation: usize) -> Option<(usize, TimeSet)> {
        let n = self.model.num_relations();
        if relation < n {
            Some((relation, TimeSet::all(self.model.num_timestamps())))
        } else {
            self.virtuals.get(relation - n).cloned()
        }
    }

    /// Time projection for a concrete fact pair: degrees over the
    /// timestamp axis for "when does `(head, relation, tail)` hold?"
    /// (TFLEX's time-projection operator, restricted to concrete anchors).
    /// Same sigmoid-temperature map as entity hops. Compose the result
    /// back into set logic via [`TimeSet::from_degrees`], then
    /// [`TimeSet::after_all`] / [`TimeSet::before_all`] for event-relative
    /// windows over a predicted (rather than known) event time. Out-of-range
    /// ids give all-zero degrees.
    pub fn when(&self, head: usize, relation: usize, tail: usize) -> Vec<f32> {
        let n = self.model.num_entities();
        if head >= n || tail >= n || relation >= self.model.num_relations() {
            return vec![0.0; self.model.num_timestamps()];
        }
        self.model
            .score_all_times(head, relation, tail)
            .iter()
            .map(|&e| sigmoid(-e / self.temperature))
            .collect()
    }
}

impl<S: tranz::temporal::TemporalScorer> AtomicScorer for TemporalPointModel<S> {
    fn num_entities(&self) -> usize {
        self.model.num_entities()
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        let n = self.model.num_entities();
        let Some((base, times)) = self.resolve(relation) else {
            return vec![0.0; n]; // unknown virtual id, as in TemporalKg.
        };
        if anchor >= n {
            return vec![0.0; n];
        }
        if times.is_empty() {
            return vec![0.0; n]; // empty set: no admissible timestamp.
        }
        // Existential over the set: the scorer folds the best (lowest)
        // energy per entity across τ ∈ S (batched and parallel where the
        // scorer overrides it), then map once (sigmoid is monotone).
        let taus: Vec<usize> = times.iter().collect();
        self.model
            .score_all_tails_over(anchor, base, &taus)
            .iter()
            .map(|&e| sigmoid(-e / self.temperature))
            .collect()
    }

    fn project_subset(&self, anchor: usize, relation: usize, candidates: &[usize]) -> Vec<f32> {
        let n = self.model.num_entities();
        let Some((base, times)) = self.resolve(relation) else {
            return vec![0.0; candidates.len()];
        };
        if anchor >= n || times.is_empty() {
            return vec![0.0; candidates.len()];
        }
        let taus: Vec<usize> = times.iter().collect();
        candidates
            .iter()
            .map(|&tail| {
                if tail >= n {
                    return 0.0;
                }
                let best = taus
                    .iter()
                    .map(|&tau| self.model.score(anchor, base, tail, tau))
                    .fold(f32::INFINITY, f32::min);
                sigmoid(-best / self.temperature)
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

    /// 3 entities, 1 relation, 4 timestamps. The fact (0, r, 1) holds only
    /// at τ = 1 (low energy); (0, r, 2) holds only at τ = 3.
    struct Planted;
    impl tranz::temporal::TemporalScorer for Planted {
        fn score(&self, h: usize, _r: usize, t: usize, tau: usize) -> f32 {
            match (h, t, tau) {
                (0, 1, 1) => -8.0,
                (0, 2, 3) => -8.0,
                _ => 8.0,
            }
        }
        fn num_entities(&self) -> usize {
            3
        }
        fn num_relations(&self) -> usize {
            1
        }
        fn num_timestamps(&self) -> usize {
            4
        }
    }

    #[test]
    fn windowed_hops_scope_answers_existentially() {
        let mut m = TemporalPointModel::new(Planted);
        let early = m.windowed(0, TimeSet::before(2, 4)).unwrap();
        let late = m.windowed(0, TimeSet::after(2, 4)).unwrap();
        let cfg = QueryConfig::default();

        let s = answer_query::<Godel>(&m, &Query::anchor(0, early), &cfg);
        assert!(s[1] > 0.99, "entity 1 holds before τ=2: {s:?}");
        assert!(s[2] < 0.01, "entity 2 does not: {s:?}");

        let s = answer_query::<Godel>(&m, &Query::anchor(0, late), &cfg);
        assert!(s[2] > 0.99 && s[1] < 0.01, "{s:?}");

        // The base relation is unconstrained: both facts admit.
        let s = answer_query::<Godel>(&m, &Query::anchor(0, 0), &cfg);
        assert!(s[1] > 0.99 && s[2] > 0.99, "{s:?}");
    }

    #[test]
    fn temporal_point_subset_matches_dense_windowed_projection() {
        let mut m = TemporalPointModel::new(Planted);
        let late = m.windowed(0, TimeSet::after(2, 4)).unwrap();
        let dense = m.project(0, late);
        assert_eq!(
            m.project_subset(0, late, &[2, 1, 99]),
            vec![dense[2], dense[1], 0.0]
        );
    }

    /// The non-contiguous set intervals cannot represent: NOT-during
    /// (complement of a between) admits both rays.
    #[test]
    fn complement_set_hops_work() {
        let mut m = TemporalPointModel::new(Planted);
        let not_mid = m
            .windowed(0, TimeSet::between(2, 2, 4).complement())
            .unwrap();
        let s = m.project(0, not_mid);
        assert!(
            s[1] > 0.99 && s[2] > 0.99,
            "τ=1 and τ=3 both admitted: {s:?}"
        );
    }

    #[test]
    fn invalid_registrations_and_ids_are_safe() {
        let mut m = TemporalPointModel::new(Planted);
        assert_eq!(m.windowed(5, TimeSet::all(4)), None, "unknown relation");
        assert_eq!(m.windowed(0, TimeSet::all(9)), None, "axis mismatch");
        let empty = m.windowed(0, TimeSet::empty(4)).unwrap();
        assert!(m.project(0, empty).iter().all(|&d| d == 0.0));
        assert!(m.project(0, 99).iter().all(|&d| d == 0.0), "unknown id");
        assert!(m.project(99, 0).iter().all(|&d| d == 0.0), "bad anchor");
    }

    /// Time projection peaks at the planted timestamp, and its extraction
    /// anchors an event-relative hop: "what does 0 reach AFTER the time
    /// (0, r, 1) held" admits only the τ=3 fact — TFLEX's Figure-1 shape,
    /// symbolically.
    #[test]
    fn when_composes_into_event_relative_hops() {
        let mut m = TemporalPointModel::new(Planted);
        let when = m.when(0, 0, 1);
        assert_eq!(when.len(), 4);
        assert!(when[1] > 0.99, "planted at τ=1: {when:?}");
        assert!(when.iter().enumerate().all(|(t, &d)| t == 1 || d < 0.01));

        let event = TimeSet::from_degrees(&when, 0.5);
        let after_event = m.windowed(0, event.after_all()).unwrap();
        let s = m.project(0, after_event);
        assert!(s[2] > 0.99, "the τ=3 fact is after the event: {s:?}");
        assert!(s[1] < 0.01, "the event fact itself is not: {s:?}");

        // Out-of-range ids are all-zero, matching project's convention.
        assert!(m.when(9, 0, 1).iter().all(|&d| d == 0.0));
        assert!(m.when(0, 9, 1).iter().all(|&d| d == 0.0));
    }

    #[test]
    fn temperature_spreads_but_preserves_ranking() {
        let sharp = TemporalPointModel::new(Planted);
        let mut soft = TemporalPointModel::with_temperature(Planted, 5.0);
        let _ = soft.windowed(0, TimeSet::all(4));
        let a = sharp.project(0, 0);
        let b = soft.project(0, 0);
        assert!(b[1] < a[1] && b[1] > 0.5, "softened but still positive");
        assert!(b[0] > a[0] && b[0] < 0.5);
    }
}
