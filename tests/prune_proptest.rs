//! Randomized differential: pruned == dense on random EPFO query trees over
//! random fuzzy graphs, across all four algebras. The strong form of the
//! "results identical, only work changes" claim: sweep random trees and
//! graphs in CI rather than one hand-built query.

use heyting::{
    answer_query_topk, answer_query_topk_pruned, FuzzyKg, Godel, Lukasiewicz, Product, Query,
    QueryConfig, Viterbi,
};
use proptest::prelude::*;

/// Random fuzzy graph with `n` entities, `nrels` relations, `edges` facts.
fn kg_strategy(n: usize, nrels: usize, edges: usize) -> impl Strategy<Value = FuzzyKg> {
    let edge = (
        0..n as u32,
        0..nrels as u32,
        0..n as u32,
        0u32..=1000, // weight*1000
    );
    prop::collection::vec(edge, edges).prop_map(move |edges| {
        let mut kg = FuzzyKg::new(n);
        for (h, r, t, w) in edges {
            kg.add_edge(h as usize, r as usize, t as usize, w as f32 / 1000.0);
        }
        kg
    })
}

/// Random EPFO query tree (leaves, projections, intersections, unions).
fn query_strategy(n: usize, nrels: usize) -> impl Strategy<Value = Query> {
    let n = n as u32;
    let nrels = nrels as u32;
    proptest::prop_oneof![
        (0..n, 0..nrels).prop_map(|(e, r)| Query::anchor(e as usize, r as usize)),
        (0..n, 0..nrels, 0..nrels).prop_map(|(e, r1, r2)| Query::Project {
            inner: Box::new(Query::anchor(e as usize, r1 as usize)),
            relation: r2 as usize,
        }),
        (0..n, 0..nrels, 0..nrels, 0..nrels).prop_map(|(e, r1, r2, r3)| Query::Project {
            inner: Box::new(Query::Project {
                inner: Box::new(Query::anchor(e as usize, r1 as usize)),
                relation: r2 as usize,
            }),
            relation: r3 as usize,
        }),
        (0..n, 0..n, 0..nrels, 0..nrels).prop_map(|(a1, a2, r1, r2)| {
            Query::intersection(vec![
                Query::anchor(a1 as usize, r1 as usize),
                Query::anchor(a2 as usize, r2 as usize),
            ])
        }),
        (0..n, 0..n, 0..nrels, 0..nrels).prop_map(|(a1, a2, r1, r2)| {
            Query::union(vec![
                Query::anchor(a1 as usize, r1 as usize),
                Query::anchor(a2 as usize, r2 as usize),
            ])
        }),
        (0..n, 0..n, 0..nrels, 0..nrels, 0..nrels).prop_map(|(a1, a2, r1, r2, r3)| {
            Query::intersection(vec![
                Query::Project {
                    inner: Box::new(Query::anchor(a1 as usize, r1 as usize)),
                    relation: r2 as usize,
                },
                Query::anchor(a2 as usize, r3 as usize),
            ])
        }),
    ]
}

fn canon(xs: &[(usize, f32)]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = xs.iter().copied().filter(|(_, d)| *d > 0.0).collect();
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    v
}

fn assert_same<T: heyting::Truth>(
    kg: &FuzzyKg,
    q: &Query,
    cfg: &QueryConfig,
) -> Result<(), proptest::test_runner::TestCaseError> {
    let dense = answer_query_topk::<T>(kg, q, cfg, 20);
    let pruned = answer_query_topk_pruned::<T>(kg, kg, q, cfg, 20);
    let a = canon(&dense);
    let b = canon(&pruned);
    if a.len() != b.len() {
        return Err(proptest::test_runner::TestCaseError::fail(format!(
            "support size differ: dense {:?}\npruned {:?}\nquery {q:?}",
            a, b
        )));
    }
    for ((ae, ad), (pe, pd)) in a.iter().zip(b.iter()) {
        if ae != pe {
            return Err(proptest::test_runner::TestCaseError::fail(format!(
                "entity mismatch {ae} vs {pe}\nquery {q:?}"
            )));
        }
        if (ad - pd).abs() > 1e-5 {
            return Err(proptest::test_runner::TestCaseError::fail(format!(
                "degree mismatch {ad} vs {pd}\nquery {q:?}"
            )));
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn pruned_matches_dense_godel(kg in kg_strategy(12, 3, 24), q in query_strategy(12, 3)) {
        assert_same::<Godel>(&kg, &q, &QueryConfig::default())?;
    }

    #[test]
    fn pruned_matches_dense_product(kg in kg_strategy(12, 3, 24), q in query_strategy(12, 3)) {
        assert_same::<Product>(&kg, &q, &QueryConfig::default())?;
    }

    #[test]
    fn pruned_matches_dense_lukasiewicz(kg in kg_strategy(12, 3, 24), q in query_strategy(12, 3)) {
        assert_same::<Lukasiewicz>(&kg, &q, &QueryConfig::default())?;
    }

    #[test]
    fn pruned_matches_dense_viterbi(kg in kg_strategy(12, 3, 24), q in query_strategy(12, 3)) {
        assert_same::<Viterbi>(&kg, &q, &QueryConfig::default())?;
    }
}
