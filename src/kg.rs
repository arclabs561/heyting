//! A reference in-memory fuzzy knowledge graph.
//!
//! [`FuzzyKg`] is the simplest [`AtomicScorer`]: a set of weighted
//! `(head, relation, tail)` edges, where `project(h, r)` returns each tail's
//! edge weight. It makes the query engine usable and testable without any
//! embedding model, and serves as the worked example of the [`AtomicScorer`]
//! contract that subsume regions and tranz points will implement.

use std::collections::HashMap;

use crate::query::AtomicScorer;

/// Edge rejected by [`FuzzyKg::try_add_edge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KgError {
    /// `id` is not below the graph's entity count.
    EntityOutOfRange {
        /// The offending entity id.
        id: usize,
        /// The graph's entity count.
        n_entities: usize,
    },
    /// The weight is NaN or infinite.
    NonFiniteWeight,
}

impl std::fmt::Display for KgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntityOutOfRange { id, n_entities } => {
                write!(
                    f,
                    "entity id {id} is out of range for {n_entities} entities"
                )
            }
            Self::NonFiniteWeight => write!(f, "edge weight is not finite"),
        }
    }
}

impl std::error::Error for KgError {}

/// An in-memory knowledge graph with fuzzy edge weights in `[0, 1]`.
///
/// Edges are keyed by `(head, relation)`; `project` looks up the tails and
/// returns their weights as membership degrees. Parallel edges to the same tail
/// take the maximum weight (the existential over edge instances).
#[derive(Debug, Clone, Default)]
pub struct FuzzyKg {
    n_entities: usize,
    edges: HashMap<(usize, usize), Vec<(usize, f32)>>,
}

impl FuzzyKg {
    /// Create an empty graph over `n_entities` entities (ids `0..n_entities`).
    pub fn new(n_entities: usize) -> Self {
        Self {
            n_entities,
            edges: HashMap::new(),
        }
    }

    /// Add a weighted edge `(head, relation, tail)`. `weight` is clamped to
    /// `[0, 1]`. Out-of-range entity ids are ignored; use
    /// [`FuzzyKg::try_add_edge`] to have them reported instead.
    pub fn add_edge(&mut self, head: usize, relation: usize, tail: usize, weight: f32) {
        if head >= self.n_entities || tail >= self.n_entities {
            return;
        }
        self.edges
            .entry((head, relation))
            .or_default()
            .push((tail, weight.clamp(0.0, 1.0)));
    }

    /// Like [`FuzzyKg::add_edge`], but returns an error instead of silently
    /// skipping an out-of-range entity id or a NaN/infinite weight. A finite
    /// weight is clamped to `[0, 1]`.
    pub fn try_add_edge(
        &mut self,
        head: usize,
        relation: usize,
        tail: usize,
        weight: f32,
    ) -> Result<(), KgError> {
        for id in [head, tail] {
            if id >= self.n_entities {
                return Err(KgError::EntityOutOfRange {
                    id,
                    n_entities: self.n_entities,
                });
            }
        }
        if !weight.is_finite() {
            return Err(KgError::NonFiniteWeight);
        }
        self.add_edge(head, relation, tail, weight);
        Ok(())
    }

    /// Number of entities.
    pub fn num_entities(&self) -> usize {
        self.n_entities
    }

    /// Number of stored edges.
    pub fn num_edges(&self) -> usize {
        self.edges.values().map(Vec::len).sum()
    }

    /// Tail entities with an edge `(anchor, relation, tail)`, unordered.
    ///
    /// This is the exact nonzero support of
    /// [`project`](crate::AtomicScorer::project), which makes `FuzzyKg` its
    /// own exact [`CandidateSource`](crate::prune::CandidateSource).
    pub fn tails(&self, anchor: usize, relation: usize) -> Vec<usize> {
        self.edges
            .get(&(anchor, relation))
            .map(|tails| tails.iter().map(|&(t, _)| t).collect())
            .unwrap_or_default()
    }
}

impl AtomicScorer for FuzzyKg {
    fn num_entities(&self) -> usize {
        self.n_entities
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        let mut scores = vec![0.0_f32; self.n_entities];
        if let Some(tails) = self.edges.get(&(anchor, relation)) {
            for &(t, w) in tails {
                if t < self.n_entities && w > scores[t] {
                    scores[t] = w;
                }
            }
        }
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An out-of-range id or a NaN weight must be reported, not dropped.
    #[test]
    fn try_add_edge_rejects_out_of_range_ids_and_nan_weights() {
        let mut kg = FuzzyKg::new(2);
        assert_eq!(
            kg.try_add_edge(0, 0, 2, 1.0),
            Err(KgError::EntityOutOfRange {
                id: 2,
                n_entities: 2
            })
        );
        assert_eq!(
            kg.try_add_edge(5, 0, 1, 1.0),
            Err(KgError::EntityOutOfRange {
                id: 5,
                n_entities: 2
            })
        );
        assert_eq!(
            kg.try_add_edge(0, 0, 1, f32::NAN),
            Err(KgError::NonFiniteWeight)
        );
        assert_eq!(kg.num_edges(), 0);
        assert_eq!(kg.try_add_edge(0, 0, 1, 2.0), Ok(()));
        assert_eq!(kg.project(0, 0), vec![0.0, 1.0]);
    }

    #[test]
    fn project_returns_edge_weights() {
        let mut kg = FuzzyKg::new(4);
        kg.add_edge(0, 0, 1, 0.9);
        kg.add_edge(0, 0, 2, 0.5);
        let s = kg.project(0, 0);
        assert_eq!(s.len(), 4);
        assert!((s[1] - 0.9).abs() < 1e-6);
        assert!((s[2] - 0.5).abs() < 1e-6);
        assert_eq!(s[3], 0.0);
    }

    #[test]
    fn parallel_edges_take_max() {
        let mut kg = FuzzyKg::new(3);
        kg.add_edge(0, 0, 1, 0.3);
        kg.add_edge(0, 0, 1, 0.8);
        assert!((kg.project(0, 0)[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn weight_is_clamped_and_oob_ignored() {
        let mut kg = FuzzyKg::new(2);
        kg.add_edge(0, 0, 1, 5.0); // clamped to 1.0
        kg.add_edge(0, 0, 9, 1.0); // out of range, ignored
        assert!((kg.project(0, 0)[1] - 1.0).abs() < 1e-6);
        assert_eq!(kg.num_edges(), 1);
    }
}
