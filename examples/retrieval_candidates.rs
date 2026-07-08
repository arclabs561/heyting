//! Data-free smoke test for the optional retrieval candidate adapters.

use heyting::adapters::{HyperbolicKnnCandidates, PrecinctSubsumers, VicinityCandidates};
use heyting::CandidateSource;
use precinct::{AxisBox, IndexParams, RegionIndex, SearchParams};

fn main() {
    let vicinity = vicinity_candidates();
    let precinct = precinct_candidates();
    let hyperbolic = hyperbolic_candidates();

    println!("vicinity candidates: {vicinity:?}");
    println!("precinct candidates: {precinct:?}");
    println!("hyperbolic candidates: {hyperbolic:?}");

    assert_eq!(vicinity[0], 1);
    assert!(precinct.contains(&0));
    assert!(precinct.contains(&1));
    assert_eq!(hyperbolic, vec![1, 0]);
}

fn vicinity_candidates() -> Vec<usize> {
    let mut index = vicinity::hnsw::HNSWIndex::builder(2)
        .metric(vicinity::DistanceMetric::L2)
        .build()
        .expect("hnsw");
    index.add_slice(0, &[0.0, 0.0]).expect("add");
    index.add_slice(1, &[1.0, 0.0]).expect("add");
    index.add_slice(2, &[3.0, 0.0]).expect("add");
    index.build().expect("build");

    let source = VicinityCandidates::new(&index, |_anchor, _relation| Some(vec![0.9, 0.0]), 2, 8);
    source.candidates(0, 0).expect("search")
}

fn precinct_candidates() -> Vec<usize> {
    let boxes = vec![
        AxisBox::new(vec![-2.0, -2.0], vec![2.0, 2.0]),
        AxisBox::new(vec![-1.0, -1.0], vec![1.0, 1.0]),
        AxisBox::new(vec![-0.25, -0.25], vec![0.25, 0.25]),
    ];
    let mut index = RegionIndex::new(2, IndexParams::default()).expect("index");
    for (id, region) in boxes.iter().cloned().enumerate() {
        index.add(id as u32, region).expect("add");
    }
    index.build().expect("build");

    let source = PrecinctSubsumers::new(&index, &boxes, 0)
        .with_min_prob(0.5)
        .with_search_params(SearchParams {
            ef: 32,
            overretrieve: 8,
        });
    source.candidates(2, 0).expect("search")
}

fn hyperbolic_candidates() -> Vec<usize> {
    let points = vec![vec![0.0, 0.0], vec![0.2, 0.0], vec![0.6, 0.0]];
    let source = HyperbolicKnnCandidates::new(
        hyperball::core::PoincareBallCore::new(1.0),
        &points,
        |_anchor, _relation| Some(vec![0.19, 0.0]),
        2,
    );
    source.candidates(0, 0).expect("search")
}
