//! Abductive hypothesis generation: the query that explains an answer set.
//!
//! Deduction asks "which entities answer this query?"; abduction flips it —
//! given observed entities, find the logical hypothesis that best explains
//! them (Bai et al., ACL 2024, "complex logical hypothesis generation").
//! This module is the symbolic version: enumerate template hypotheses
//! (one-hop atoms and their conjunctions), score each against the
//! observation by fuzzy Jaccard similarity, return the best.
//!
//! It is the abductive twin of [`crate::provenance`]: a witness explains one
//! answer of a known query, an abduced hypothesis explains a whole answer
//! set with no query given. Combined with `explain_answer`, a recovered
//! hypothesis is itself explainable.
//!
//! Cost is `|anchors| × |relations|` one-hop projections for the atom sweep
//! plus `beam²` cheap vector combinations for conjunctions; restrict
//! [`AbduceConfig::anchors`] for large embedding scorers.

use crate::query::{answer_query, AtomicScorer, Query, QueryConfig};
use crate::truth::Truth;

/// Search-space and beam knobs for [`abduce`].
#[derive(Debug, Clone)]
pub struct AbduceConfig {
    /// Anchor entities to consider for hypothesis atoms; `None` = all.
    pub anchors: Option<Vec<usize>>,
    /// Maximum conjuncts per hypothesis (1 = atoms only, 2 = pairs). Values
    /// above 2 are treated as 2 in this template-based v1.
    pub max_conjuncts: usize,
    /// Atoms kept for conjunction pairing, and hypotheses returned.
    pub beam: usize,
    /// Evaluation knobs for scoring hypotheses.
    pub query: QueryConfig,
}

impl Default for AbduceConfig {
    fn default() -> Self {
        Self {
            anchors: None,
            max_conjuncts: 2,
            beam: 16,
            query: QueryConfig::default(),
        }
    }
}

/// A hypothesis with its explanation quality.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// The abduced query.
    pub query: Query,
    /// Fuzzy Jaccard between the query's answer degrees and the observed
    /// set: `Σ min(deg, obs) / Σ max(deg, obs)` with `obs ∈ {0, 1}`. `1.0`
    /// is a perfect explanation (full coverage, no excess mass).
    pub score: f32,
}

/// Abduce the template hypotheses best explaining `observed`.
///
/// Sweeps one-hop atoms `(anchor, relation)` over the configured space,
/// keeps the top [`AbduceConfig::beam`] by fuzzy Jaccard against the
/// observation, then (when `max_conjuncts >= 2`) scores pairwise
/// conjunctions of kept atoms in the algebra `T`. Returns up to `beam`
/// hypotheses, best first. Empty `observed` or `relations` returns none.
pub fn abduce<T: Truth>(
    scorer: &dyn AtomicScorer,
    observed: &[usize],
    relations: &[usize],
    config: &AbduceConfig,
) -> Vec<Hypothesis> {
    let n = scorer.num_entities();
    if observed.is_empty() || relations.is_empty() || n == 0 {
        return vec![];
    }
    let mut obs = vec![0.0_f32; n];
    for &e in observed {
        if e < n {
            obs[e] = 1.0;
        }
    }

    // Atom sweep.
    let anchors: Vec<usize> = match &config.anchors {
        Some(a) => a.clone(),
        None => (0..n).collect(),
    };
    let mut atoms: Vec<(Query, Vec<f32>, f32)> = Vec::new();
    for &h in &anchors {
        for &r in relations {
            let q = Query::anchor(h, r);
            let degrees = answer_query::<T>(scorer, &q, &config.query);
            let score = fuzzy_jaccard(&degrees, &obs);
            if score > 0.0 {
                atoms.push((q, degrees, score));
            }
        }
    }
    atoms.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    atoms.truncate(config.beam);

    let mut out: Vec<Hypothesis> = atoms
        .iter()
        .map(|(q, _, s)| Hypothesis {
            query: q.clone(),
            score: *s,
        })
        .collect();

    // Pairwise conjunctions of kept atoms, combined in the algebra.
    if config.max_conjuncts >= 2 {
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let degrees: Vec<f32> = atoms[i]
                    .1
                    .iter()
                    .zip(atoms[j].1.iter())
                    .map(|(&a, &b)| T::and(a, b))
                    .collect();
                let score = fuzzy_jaccard(&degrees, &obs);
                if score > 0.0 {
                    out.push(Hypothesis {
                        query: Query::intersection(vec![atoms[i].0.clone(), atoms[j].0.clone()]),
                        score,
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(config.beam);
    out
}

/// Fuzzy Jaccard `Σ min / Σ max` between a degree vector and a crisp set.
fn fuzzy_jaccard(degrees: &[f32], obs: &[f32]) -> f32 {
    let (mut inter, mut union) = (0.0_f32, 0.0_f32);
    for (d, o) in degrees.iter().zip(obs.iter()) {
        inter += d.min(*o);
        union += d.max(*o);
    }
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::FuzzyKg;
    use crate::truth::Godel;

    /// 0=animal 1=mammal 2=dog 3=cat 4=fish 5=plant; 0=is_a, 1=eats.
    fn kg() -> FuzzyKg {
        let mut kg = FuzzyKg::new(6);
        kg.add_edge(2, 0, 1, 1.0); // dog is_a mammal
        kg.add_edge(3, 0, 1, 1.0); // cat is_a mammal
        kg.add_edge(1, 0, 0, 1.0); // mammal is_a animal
        kg.add_edge(2, 1, 4, 1.0); // dog eats fish
        kg.add_edge(3, 1, 4, 1.0); // cat eats fish
        kg.add_edge(3, 1, 5, 1.0); // cat eats plant
        kg
    }

    /// Observing exactly the answers of a planted atom recovers it with a
    /// perfect score.
    #[test]
    fn recovers_a_planted_atom() {
        let kg = kg();
        // Observed: everything dog eats = {fish}.
        let hyps = abduce::<Godel>(&kg, &[4], &[0, 1], &AbduceConfig::default());
        let best = &hyps[0];
        assert!((best.score - 1.0).abs() < 1e-6, "score {}", best.score);
        // Both (dog eats ?) and (cat... no: cat eats {fish, plant} over-covers.
        // dog eats {fish} is exact.
        match &best.query {
            Query::Anchor { entity, relation } => {
                assert_eq!((*entity, *relation), (2, 1), "expected dog-eats");
            }
            other => panic!("expected an atom, got {other:?}"),
        }
    }

    /// A conjunctive observation is explained best by the conjunction, not
    /// by either conjunct alone: {things that are mammals AND eat fish
    /// according to someone} — observed {fish} ∩ ... use a planted 2i:
    /// answers of (dog eats ?) AND (cat eats ?) = {fish}, while cat-eats
    /// alone over-covers with plant.
    #[test]
    fn conjunction_beats_overbroad_atoms() {
        let kg = kg();
        // Observed: {fish, plant} = exactly cat-eats; the atom wins over any
        // conjunction (which would drop plant).
        let hyps = abduce::<Godel>(&kg, &[4, 5], &[1], &AbduceConfig::default());
        match &hyps[0].query {
            Query::Anchor { entity, relation } => {
                assert_eq!((*entity, *relation), (3, 1), "expected cat-eats");
            }
            other => panic!("expected cat-eats atom, got {other:?}"),
        }
        assert!((hyps[0].score - 1.0).abs() < 1e-6);
    }

    /// Noisy observations degrade the score but keep the right hypothesis
    /// on top.
    #[test]
    fn noise_degrades_score_not_ranking() {
        let kg = kg();
        // {fish} plus an unexplainable extra (animal).
        let hyps = abduce::<Godel>(&kg, &[4, 0], &[1], &AbduceConfig::default());
        let best = &hyps[0];
        assert!(best.score < 1.0);
        match &best.query {
            Query::Anchor { entity, relation } => {
                assert_eq!((*entity, *relation), (2, 1), "dog-eats still best");
            }
            other => panic!("expected atom, got {other:?}"),
        }
    }

    #[test]
    fn empty_inputs_yield_no_hypotheses() {
        let kg = kg();
        assert!(abduce::<Godel>(&kg, &[], &[0], &AbduceConfig::default()).is_empty());
        assert!(abduce::<Godel>(&kg, &[4], &[], &AbduceConfig::default()).is_empty());
    }
}
