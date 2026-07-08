//! Candidate-source adapters for retrieval crates.

use crate::prune::CandidateSource;

/// Candidate source backed by a `vicinity` HNSW index.
///
/// The index stores tail/entity vectors. The query builder maps each
/// `(anchor, relation)` hop to the vector that should be searched, for example
/// `head + relation` for a TransE-style scorer. Returned `vicinity` document
/// IDs are interpreted as `heyting` entity IDs.
#[cfg(feature = "vicinity")]
pub struct VicinityCandidates<'a, F> {
    index: &'a vicinity::hnsw::HNSWIndex,
    query: F,
    k: usize,
    ef: usize,
}

#[cfg(feature = "vicinity")]
impl<'a, F> VicinityCandidates<'a, F> {
    /// Create a `vicinity`-backed candidate source.
    pub fn new(index: &'a vicinity::hnsw::HNSWIndex, query: F, k: usize, ef: usize) -> Self {
        Self {
            index,
            query,
            k,
            ef,
        }
    }

    /// Number of candidates requested per hop.
    pub fn k(&self) -> usize {
        self.k
    }

    /// HNSW search width.
    pub fn ef(&self) -> usize {
        self.ef
    }
}

#[cfg(feature = "vicinity")]
impl<F> CandidateSource for VicinityCandidates<'_, F>
where
    F: Fn(usize, usize) -> Option<Vec<f32>>,
{
    fn candidates(&self, anchor: usize, relation: usize) -> Option<Vec<usize>> {
        let query = (self.query)(anchor, relation)?;
        self.index.search(&query, self.k, self.ef).ok().map(|hits| {
            hits.into_iter()
                .filter_map(|(id, _)| usize::try_from(id).ok())
                .collect()
        })
    }
}

/// Candidate source for hypernym/subsumption hops backed by `precinct`.
///
/// The `RegionIndex` stores regions under external IDs. This adapter assumes
/// those IDs are the same integers used by the `AtomicScorer` entity axis. The
/// separate `regions` slice supplies the anchor region for each entity ID.
#[cfg(feature = "precinct")]
pub struct PrecinctSubsumers<'a, R: precinct::Region> {
    index: &'a precinct::RegionIndex<R>,
    regions: &'a [R],
    relation: usize,
    min_prob: f32,
    ef: usize,
    overretrieve: usize,
    exclude_anchor: bool,
}

#[cfg(feature = "precinct")]
impl<'a, R: precinct::Region> PrecinctSubsumers<'a, R> {
    /// Create a source for one subsumption relation.
    pub fn new(index: &'a precinct::RegionIndex<R>, regions: &'a [R], relation: usize) -> Self {
        let params = precinct::SearchParams::default();
        Self {
            index,
            regions,
            relation,
            min_prob: 0.0,
            ef: params.ef,
            overretrieve: params.overretrieve,
            exclude_anchor: true,
        }
    }

    /// Set the minimum soft-subsumption probability.
    pub fn with_min_prob(mut self, min_prob: f32) -> Self {
        self.min_prob = if min_prob.is_finite() {
            min_prob.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    /// Set the `precinct` search parameters.
    pub fn with_search_params(mut self, params: precinct::SearchParams) -> Self {
        self.ef = params.ef;
        self.overretrieve = params.overretrieve;
        self
    }

    /// Include the anchor entity in its own candidate set.
    pub fn include_anchor(mut self) -> Self {
        self.exclude_anchor = false;
        self
    }

    fn params(&self) -> precinct::SearchParams {
        precinct::SearchParams {
            ef: self.ef,
            overretrieve: self.overretrieve,
        }
    }
}

#[cfg(feature = "precinct")]
impl<R: precinct::Region> CandidateSource for PrecinctSubsumers<'_, R> {
    fn candidates(&self, anchor: usize, relation: usize) -> Option<Vec<usize>> {
        if relation != self.relation || anchor >= self.regions.len() {
            return Some(Vec::new());
        }

        self.index
            .subsumers_soft(&self.regions[anchor], self.min_prob, self.params())
            .ok()
            .map(|hits| {
                hits.into_iter()
                    .filter_map(|(id, _)| usize::try_from(id).ok())
                    .filter(|&id| !self.exclude_anchor || id != anchor)
                    .collect()
            })
    }
}

/// Brute-force hyperbolic nearest-neighbor candidate source.
///
/// This is useful for small hyperbolic embedding sets or as an exact oracle for
/// an ANN-backed hyperbolic source. The query builder maps `(anchor, relation)`
/// to a point in the Poincaré ball; the nearest stored entity points become the
/// candidate set.
#[cfg(feature = "hyperball")]
pub struct HyperbolicKnnCandidates<'a, F> {
    ball: hyperball::core::PoincareBallCore<f32>,
    points: &'a [Vec<f32>],
    query: F,
    k: usize,
}

#[cfg(feature = "hyperball")]
impl<'a, F> HyperbolicKnnCandidates<'a, F> {
    /// Create a hyperbolic k-NN candidate source.
    pub fn new(
        ball: hyperball::core::PoincareBallCore<f32>,
        points: &'a [Vec<f32>],
        query: F,
        k: usize,
    ) -> Self {
        Self {
            ball,
            points,
            query,
            k,
        }
    }

    /// Number of candidates returned per hop.
    pub fn k(&self) -> usize {
        self.k
    }
}

#[cfg(feature = "hyperball")]
impl<F> CandidateSource for HyperbolicKnnCandidates<'_, F>
where
    F: Fn(usize, usize) -> Option<Vec<f32>>,
{
    fn candidates(&self, anchor: usize, relation: usize) -> Option<Vec<usize>> {
        let query = (self.query)(anchor, relation)?;
        let mut scored: Vec<(usize, f32)> = self
            .points
            .iter()
            .enumerate()
            .filter(|(_, point)| point.len() == query.len())
            .map(|(id, point)| (id, self.ball.distance(&query, point)))
            .collect();
        scored.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(self.k);
        Some(scored.into_iter().map(|(id, _)| id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hyperball")]
    #[test]
    fn hyperbolic_candidates_return_nearest_points() {
        let points = vec![vec![0.0, 0.0], vec![0.2, 0.0], vec![0.6, 0.0]];
        let source = HyperbolicKnnCandidates::new(
            hyperball::core::PoincareBallCore::new(1.0),
            &points,
            |_anchor, _relation| Some(vec![0.19, 0.0]),
            2,
        );

        assert_eq!(source.candidates(0, 0), Some(vec![1, 0]));
    }

    #[cfg(feature = "precinct")]
    #[test]
    fn precinct_subsumers_return_soft_ancestors() {
        use precinct::{AxisBox, IndexParams, RegionIndex, SearchParams};

        let boxes = vec![
            AxisBox::new(vec![-2.0, -2.0], vec![2.0, 2.0]),
            AxisBox::new(vec![-1.0, -1.0], vec![1.0, 1.0]),
            AxisBox::new(vec![-0.25, -0.25], vec![0.25, 0.25]),
        ];
        let mut index = RegionIndex::new(2, IndexParams::default()).unwrap();
        for (id, region) in boxes.iter().cloned().enumerate() {
            index.add(id as u32, region).unwrap();
        }
        index.build().unwrap();

        let source = PrecinctSubsumers::new(&index, &boxes, 0)
            .with_min_prob(0.5)
            .with_search_params(SearchParams {
                ef: 32,
                overretrieve: 8,
            });
        let candidates = source.candidates(2, 0).unwrap();
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&1));
        assert!(!candidates.contains(&2));
    }

    #[cfg(feature = "vicinity")]
    #[test]
    fn vicinity_candidates_search_hnsw() {
        let mut index = vicinity::hnsw::HNSWIndex::builder(2)
            .metric(vicinity::DistanceMetric::L2)
            .build()
            .unwrap();
        index.add_slice(0, &[0.0, 0.0]).unwrap();
        index.add_slice(1, &[1.0, 0.0]).unwrap();
        index.add_slice(2, &[3.0, 0.0]).unwrap();
        index.build().unwrap();

        let source =
            VicinityCandidates::new(&index, |_anchor, _relation| Some(vec![0.9, 0.0]), 2, 8);
        assert_eq!(source.candidates(0, 0).unwrap()[0], 1);
    }
}
