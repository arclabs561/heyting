//! Why a higher-scoring false positive can disappear from a filtered ranking.
//! Run with `cargo run --example negated_answer_filter`.

use heyting::{crisp_answers, hard_answer_metrics, split_answers, FuzzyKg, Query, QueryAnswers};
use std::collections::BTreeSet;

fn main() {
    let (ada, bea, host) = (0, 1, 2);
    let (invited, declined) = (0, 1);
    let mut train = FuzzyKg::new(3);
    train.add_edge(host, invited, ada, 1.0);

    // Two facts arrive. No edge is deleted.
    let mut full = train.clone();
    full.add_edge(host, declined, ada, 1.0);
    full.add_edge(host, invited, bea, 1.0);
    let attending = Query::intersection(vec![
        Query::anchor(host, invited),
        Query::anchor(host, declined).negate(),
    ]);

    let benchmark = split_answers(&train, &full, &attending, 0.5);
    let current = crisp_answers(&full, &attending, 0.5);
    assert_eq!(benchmark.easy, BTreeSet::from([ada]));
    assert_eq!(benchmark.hard, BTreeSet::from([bea]));
    assert_eq!(current, BTreeSet::from([bea]));

    // Hold the target and predictions fixed. Change only the filter.
    let scores = [0.9, 0.5, 0.0]; // Ada, Bea, host
    let filtered = hard_answer_metrics(&scores, &benchmark);
    let current_only = QueryAnswers {
        easy: benchmark.easy.intersection(&current).copied().collect(),
        hard: benchmark.hard,
    };
    let diagnostic = hard_answer_metrics(&scores, &current_only);
    assert_eq!((filtered.mrr, diagnostic.mrr), (1.0, 0.5));

    println!("Invited with no recorded decline: initially Ada, now Bea.");
    println!("Same predictions: Ada 0.9, Bea 0.5, host 0.0.");
    println!(
        "BetaE filter removes Ada: Bea ranks first (MRR {:.1}).",
        filtered.mrr
    );
    println!(
        "Full-graph filter keeps Ada in the ranking: Bea ranks second (MRR {:.1}).",
        diagnostic.mrr
    );
}
