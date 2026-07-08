//! Public-API fixture for the Viterbi/QTO recurrence.

use heyting::{answer_query, explain_answer, FuzzyKg, Query, QueryConfig, Viterbi, Witness};

#[test]
fn viterbi_chain_matches_hand_computed_qto_recurrence() {
    let mut kg = FuzzyKg::new(6);
    kg.add_edge(0, 0, 1, 0.5);
    kg.add_edge(0, 0, 2, 0.8);
    kg.add_edge(0, 0, 3, 0.1);
    kg.add_edge(1, 1, 4, 0.7);
    kg.add_edge(2, 1, 4, 0.4);
    kg.add_edge(2, 1, 5, 0.9);
    kg.add_edge(3, 1, 5, 1.0);

    let query = Query::anchor(0, 0).then(1);
    let degrees = answer_query::<Viterbi>(&kg, &query, &QueryConfig::exact());

    let expected = [0.0, 0.0, 0.0, 0.0, 0.35, 0.72];
    for (actual, expected) in degrees.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "actual {actual}, expected {expected}"
        );
    }

    let witness = explain_answer::<Viterbi>(&kg, &query, &QueryConfig::exact(), 5).unwrap();
    assert!((witness.degree() - 0.72).abs() < 1e-6);
    match witness {
        Witness::Via {
            intermediate,
            degree,
            ..
        } => {
            assert_eq!(intermediate, 2);
            assert!((degree - 0.72).abs() < 1e-6);
        }
        other => panic!("expected via witness, got {other:?}"),
    }
}
