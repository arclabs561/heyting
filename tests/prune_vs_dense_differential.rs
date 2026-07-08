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
