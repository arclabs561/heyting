//! Time-scoped complex queries over a temporal knowledge graph.
//!
//! Facts carry validity intervals; a [`TimeWindow`] registered on the graph
//! becomes a virtual relation id, so temporal operators (before / after /
//! between) compose through the ordinary query connectives — including the
//! query this example centers on: "who held the office before 1990 AND
//! after 2000", an intersection of two windowed hops that only an entity
//! with two separate terms satisfies.
//!
//! Run: cargo run --example temporal_query

use heyting::{answer_query_topk, Godel, Query, QueryConfig, TemporalKg, TimeWindow};

fn main() {
    // Entities.
    const ALICE: usize = 0;
    const BOB: usize = 1;
    const CAROL: usize = 2;
    const OFFICE: usize = 3;
    const PARTY_X: usize = 4;
    let names = ["alice", "bob", "carol", "office", "party_x"];

    // Base relations.
    const HOLDS: usize = 0; // (office, holds, person) over a term interval
    const MEMBER: usize = 1; // (party, member, person)

    let mut kg = TemporalKg::new(5, 2);
    kg.add_fact(OFFICE, HOLDS, ALICE, 1993.0, 2001.0, 1.0);
    kg.add_fact(OFFICE, HOLDS, BOB, 1985.0, 1989.0, 1.0); // first term
    kg.add_fact(OFFICE, HOLDS, BOB, 2005.0, 2009.0, 1.0); // second term
    kg.add_fact(OFFICE, HOLDS, CAROL, 2017.0, 2021.0, 1.0);
    kg.add_fact(PARTY_X, MEMBER, BOB, 1980.0, 2010.0, 0.9);
    kg.add_fact(PARTY_X, MEMBER, CAROL, 2015.0, 2022.0, 0.9);

    // Temporal operators become virtual relation ids.
    let holds_before_1990 = kg.windowed(HOLDS, TimeWindow::Before(1990.0)).unwrap();
    let holds_after_2000 = kg.windowed(HOLDS, TimeWindow::After(2000.0)).unwrap();
    let holds_in_90s = kg
        .windowed(HOLDS, TimeWindow::Between(1990.0, 1999.9))
        .unwrap();

    let cfg = QueryConfig::default();
    let show = |label: &str, q: &Query| {
        let top = answer_query_topk::<Godel>(&kg, q, &cfg, 3);
        let pretty: Vec<String> = top
            .iter()
            .filter(|(_, d)| *d > 0.0)
            .map(|(e, d)| format!("{} ({d:.2})", names[*e]))
            .collect();
        println!("{label:<55} -> {}", pretty.join(", "));
    };

    show(
        "held office in the 1990s",
        &Query::anchor(OFFICE, holds_in_90s),
    );
    show(
        "held office before 1990 AND after 2000 (two terms)",
        &Query::intersection(vec![
            Query::anchor(OFFICE, holds_before_1990),
            Query::anchor(OFFICE, holds_after_2000),
        ]),
    );
    show(
        "held office after 2000 AND member of party_x",
        &Query::intersection(vec![
            Query::anchor(OFFICE, holds_after_2000),
            Query::anchor(PARTY_X, MEMBER),
        ]),
    );
}
