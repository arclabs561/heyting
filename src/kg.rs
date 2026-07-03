//! A reference in-memory fuzzy knowledge graph.
//!
//! [`FuzzyKg`] is the simplest [`AtomicScorer`]: a set of weighted
//! `(head, relation, tail)` edges, where `project(h, r)` returns each tail's
//! edge weight. It makes the query engine usable and testable without any
//! embedding model, and serves as the worked example of the [`AtomicScorer`]
//! contract that subsume regions and tranz points will implement.

use std::collections::HashMap;

use crate::query::AtomicScorer;

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
    /// `[0, 1]`. Out-of-range entity ids are ignored.
    pub fn add_edge(&mut self, head: usize, relation: usize, tail: usize, weight: f32) {
        if head >= self.n_entities || tail >= self.n_entities {
            return;
        }
        self.edges
            .entry((head, relation))
            .or_default()
            .push((tail, weight.clamp(0.0, 1.0)));
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
