//! Public-API differential test for exact candidate pruning.

use heyting::{answer_query_topk, answer_query_topk_pruned, FuzzyKg, Product, Query, QueryConfig};

#[test]
fn exact_candidate_pruning_matches_dense_product_chain() {
    let mut kg = FuzzyKg::new(12);
    for intermediate in 1..8 {
        kg.add_edge(0, 0, intermediate, 1.0 - intermediate as f32 * 0.05);
    }
    kg.add_edge(1, 1, 8, 0.9);
    kg.add_edge(2, 1, 8, 0.7);
    kg.add_edge(2, 1, 9, 0.8);
    kg.add_edge(3, 1, 9, 0.85);
    kg.add_edge(4, 1, 10, 0.6);
    kg.add_edge(5, 1, 10, 0.9);
    kg.add_edge(6, 1, 11, 0.7);
    kg.add_edge(7, 1, 11, 0.95);

    let query = Query::anchor(0, 0).then(1);
    let config = QueryConfig::exact();
    let dense = answer_query_topk::<Product>(&kg, &query, &config, 6);
    let pruned = answer_query_topk_pruned::<Product>(&kg, &kg, &query, &config, 6);

    assert_eq!(
        pruned,
        dense
            .into_iter()
            .filter(|(_, degree)| *degree > 0.0)
            .collect::<Vec<_>>()
    );
}

/// Hand oracle: the only path to entity 3 uses intermediate 2, giving
/// max(and(1, 0), and(0.8, 1)) = 0.8. Beam 1 selects intermediate 1 instead.
#[test]
fn negated_chain_dense_fallback_preserves_beam_config() {
    use heyting::{answer_query, Lukasiewicz};

    let mut kg = FuzzyKg::new(4);
    kg.try_add_edge(0, 0, 1, 1.0).unwrap();
    kg.try_add_edge(0, 0, 2, 0.8).unwrap();
    kg.try_add_edge(2, 1, 3, 1.0).unwrap();
    let query = Query::anchor(0, 0).then(1);
    let beam = QueryConfig { beam_k: 1 };
    let exact = QueryConfig::exact();
    assert_eq!(answer_query::<Lukasiewicz>(&kg, &query, &beam)[3], 0.0);
    assert!((answer_query::<Lukasiewicz>(&kg, &query, &exact)[3] - 0.8).abs() < 1e-6);

    let negated = query.negate();
    assert_eq!(answer_query::<Lukasiewicz>(&kg, &negated, &beam)[3], 1.0);
    assert!((answer_query::<Lukasiewicz>(&kg, &negated, &exact)[3] - 0.2).abs() < 1e-6);
    for config in [&beam, &exact] {
        assert_eq!(
            answer_query_topk_pruned::<Lukasiewicz>(&kg, &kg, &negated, config, 4),
            answer_query_topk::<Lukasiewicz>(&kg, &negated, config, 4),
        );
    }
}
