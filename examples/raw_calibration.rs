//! Calibrate raw one-hop scores before logical query evaluation.
//!
//! Usage: cargo run --example raw_calibration

use heyting::{
    answer_query_topk, AffineSigmoidCalibrator, AtomicScorer, CalibratedScorer, Godel, Query,
    QueryConfig, RawProjection, RawScoreOrder,
};

const MAMMAL: usize = 1;
const DOG: usize = 2;
const CAT: usize = 3;
const IS_A: usize = 0;
const NAMES: [&str; 5] = ["animal", "mammal", "dog", "cat", "plant"];

struct RawTaxonomyScorer;

impl AtomicScorer for RawTaxonomyScorer {
    fn num_entities(&self) -> usize {
        NAMES.len()
    }

    fn project(&self, _anchor: usize, _relation: usize) -> Vec<f32> {
        vec![0.0; NAMES.len()]
    }

    fn project_raw(&self, anchor: usize, relation: usize) -> Option<RawProjection> {
        if relation != IS_A {
            return None;
        }

        let scores = match anchor {
            DOG | CAT => vec![1.0, 0.0, 3.0, 3.0, 4.0],
            MAMMAL => vec![0.0, 2.0, 4.0, 4.0, 5.0],
            _ => vec![4.0; NAMES.len()],
        };
        Some(RawProjection::new(scores, RawScoreOrder::LowerIsBetter))
    }
}

fn main() {
    let scorer = CalibratedScorer::new(RawTaxonomyScorer, AffineSigmoidCalibrator::new(4.0, 2.0));
    let query = Query::intersection(vec![Query::anchor(DOG, IS_A), Query::anchor(CAT, IS_A)]);
    let top = answer_query_topk::<Godel>(&scorer, &query, &QueryConfig::default(), 3);

    println!("(dog is_a ?) AND (cat is_a ?)");
    for (rank, (entity, degree)) in top.iter().enumerate() {
        println!("  #{} {:<8} {:.3}", rank + 1, NAMES[*entity], degree);
    }

    assert_eq!(top[0].0, MAMMAL);
    assert!(top[0].1 > top[1].1);
}
