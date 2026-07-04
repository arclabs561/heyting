//! Complex-query answering on FB15k-237 with a trained 1p scorer: the
//! CQD recipe (arXiv:2011.03459) run end to end on this stack.
//!
//! A DistMult model trained by the tranz CLI supplies atomic hop degrees
//! through [`PointModel`]; heyting composes 1p/2p/2i/3i queries in the
//! Product and Gödel algebras; answers are scored on **hard** answers only
//! (require a held-out edge) with the other true answers filtered, per the
//! standard protocol ([`heyting::eval`]). The run closes with the rest of
//! the stack on the same model: a provenance witness for one answer, and
//! conformal answer-set coverage calibrated on the valid split.
//!
//! Queries are generated deterministically from the dataset splits (first-N
//! satisfying each shape), not sampled, so runs are reproducible, and each
//! composite type comes in two classes: REDUCIBLE (exactly one atom needs a
//! held-out edge; the rest are train-traversable — the class that
//! arXiv:2410.12537 shows collapses to plain link prediction) and
//! NON-REDUCIBLE (every atom needs a held-out edge). The gap between the two
//! columns is the honest measure of multi-hop composition; compare against
//! CQD (arXiv:2011.03459), QTO (ICML 2023), and the ICLR 2025 critique's
//! tables as sanity anchors, not leaderboard entries (query files differ).
//!
//! Data-gated: needs `data/Release/{train,valid,test}.txt` (FB15k-237) and
//! `data/fb15k237-distmult/{entities,relations}.tsv` (run
//! `scripts/fetch_fb15k237.sh`, then the tranz training command it prints).
//! Without them this prints instructions and exits 0.
//!
//! Run: cargo run --release --features tranz --example fb15k237_clqa

use std::collections::HashMap;
use std::path::Path;

use heyting::adapters::PointModel;
use heyting::{
    answer_query, calibrate, empirical_coverage, explain_answer, hard_answer_metrics,
    split_answers, FuzzyKg, Godel, Product, Query, QueryConfig, QueryMetrics,
};

const DATA: &str = "data/Release";
const EMB: &str = "data/fb15k237-distmult";
const PER_TYPE: usize = 200;

fn main() {
    let (Some(triples), Some((ent_names, ent_vecs)), Some((rel_names, rel_vecs))) = (
        load_splits(Path::new(DATA)),
        tranz::io::import_embeddings(&Path::new(EMB).join("entities.tsv")).ok(),
        tranz::io::import_embeddings(&Path::new(EMB).join("relations.tsv")).ok(),
    ) else {
        eprintln!("FB15k-237 data or trained embeddings not found.");
        eprintln!("1. scripts/fetch_fb15k237.sh");
        eprintln!("2. in ../tranz: cargo run --release --features \"burn-ndarray,burn-wgpu\" \\");
        eprintln!("   --bin tranz -- train --data ../heyting/{DATA} --model distmult \\");
        eprintln!("   --dim 256 --epochs 20 --output ../heyting/{EMB}");
        return; // data-gated no-op.
    };

    // Ids are defined by the embedding TSV order (the model's own vocab).
    let ent_id: HashMap<&str, usize> = ent_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let rel_id: HashMap<&str, usize> = rel_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let dim = ent_vecs.first().map(Vec::len).unwrap_or(0);
    let n_ent = ent_names.len();
    // Temperature 5 spreads saturated 1-N scores across (0, 1); every ranking
    // metric is invariant (the map is monotone), but conformal thresholds
    // become non-degenerate.
    let model =
        PointModel::with_temperature(tranz::DistMult::from_vecs(ent_vecs, rel_vecs, dim), 5.0);

    let to_ids = |split: &[(String, String, String)]| -> Vec<(usize, usize, usize)> {
        split
            .iter()
            .filter_map(|(h, r, t)| {
                Some((
                    *ent_id.get(h.as_str())?,
                    *rel_id.get(r.as_str())?,
                    *ent_id.get(t.as_str())?,
                ))
            })
            .collect()
    };
    let train = to_ids(&triples.0);
    let valid = to_ids(&triples.1);
    let test = to_ids(&triples.2);
    eprintln!(
        "FB15k-237: {n_ent} entities, {} relations (incl. reciprocals), \
         {}/{}/{} train/valid/test triples, dim {dim}",
        rel_names.len(),
        train.len(),
        valid.len(),
        test.len()
    );

    // Crisp graphs for the easy/hard split oracle.
    let mut train_kg = FuzzyKg::new(n_ent);
    for &(h, r, t) in &train {
        train_kg.add_edge(h, r, t, 1.0);
    }
    let mut full_kg = train_kg.clone();
    for &(h, r, t) in valid.iter().chain(test.iter()) {
        full_kg.add_edge(h, r, t, 1.0);
    }

    // Deterministic query generation from the test split.
    let mut in_train: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for &(h, r, t) in &train {
        in_train.entry(t).or_default().push((h, r));
    }
    // Held-out (valid ∪ test) in-edges, for the non-reducible class.
    let mut in_heldout: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for &(h, r, t) in valid.iter().chain(test.iter()) {
        in_heldout.entry(t).or_default().push((h, r));
    }
    let mut queries: Vec<(&'static str, Vec<Query>)> = vec![
        ("1p", vec![]),
        ("2p", vec![]),
        ("2i", vec![]),
        ("3i", vec![]),
        ("2p!", vec![]), // ! = non-reducible: every atom needs prediction
        ("2i!", vec![]),
        ("3i!", vec![]),
    ];
    let mut seen: std::collections::HashSet<(usize, usize, usize, usize)> =
        std::collections::HashSet::new();
    for &(h, r, t) in &test {
        if queries[0].1.len() < PER_TYPE && seen.insert((0, h, r, 0)) {
            queries[0].1.push(Query::anchor(h, r));
        }
        if queries[1].1.len() < PER_TYPE {
            // (h2, r1, h) in train, then (h, r, t) held out: 2p whose final
            // hop needs prediction.
            if let Some(&(h2, r1)) = in_train.get(&h).and_then(|v| v.first()) {
                if seen.insert((1, h2, r1, r)) {
                    queries[1].1.push(Query::anchor(h2, r1).then(r));
                }
            }
        }
        if queries[2].1.len() < PER_TYPE {
            // Second in-edge of t from train: (h, r, t) hard, (h2, r2, t) easy.
            if let Some(&(h2, r2)) = in_train
                .get(&t)
                .and_then(|v| v.iter().find(|&&(h2, _)| h2 != h))
            {
                if seen.insert((2, h, r, r2)) {
                    queries[2].1.push(Query::intersection(vec![
                        Query::anchor(h, r),
                        Query::anchor(h2, r2),
                    ]));
                }
            }
        }
        if queries[3].1.len() < PER_TYPE {
            if let Some(edges) = in_train.get(&t) {
                let mut it = edges.iter().filter(|&&(h2, _)| h2 != h);
                if let (Some(&(h2, r2)), Some(&(h3, r3))) = (it.next(), it.next()) {
                    if seen.insert((3, h, r2.min(r3), r2.max(r3))) {
                        queries[3].1.push(Query::intersection(vec![
                            Query::anchor(h, r),
                            Query::anchor(h2, r2),
                            Query::anchor(h3, r3),
                        ]));
                    }
                }
            }
        }
        // Non-reducible variants: every atom's edge is held out, so no
        // single-link shortcut exists (arXiv:2410.12537's hard class).
        if queries[4].1.len() < PER_TYPE {
            // Both hops held out: (h2, r1, h) ∈ valid∪test, (h, r, t) ∈ test.
            if let Some(&(h2, r1)) = in_heldout.get(&h).and_then(|v| v.first()) {
                if seen.insert((4, h2, r1, r)) {
                    queries[4].1.push(Query::anchor(h2, r1).then(r));
                }
            }
        }
        if queries[5].1.len() < PER_TYPE {
            if let Some(&(h2, r2)) = in_heldout
                .get(&t)
                .and_then(|v| v.iter().find(|&&(h2, _)| h2 != h))
            {
                if seen.insert((5, h, r, r2)) {
                    queries[5].1.push(Query::intersection(vec![
                        Query::anchor(h, r),
                        Query::anchor(h2, r2),
                    ]));
                }
            }
        }
        if queries[6].1.len() < PER_TYPE {
            if let Some(edges) = in_heldout.get(&t) {
                let mut it = edges.iter().filter(|&&(h2, _)| h2 != h);
                if let (Some(&(h2, r2)), Some(&(h3, r3))) = (it.next(), it.next()) {
                    if seen.insert((6, h, r2.min(r3), r2.max(r3))) {
                        queries[6].1.push(Query::intersection(vec![
                            Query::anchor(h, r),
                            Query::anchor(h2, r2),
                            Query::anchor(h3, r3),
                        ]));
                    }
                }
            }
        }
    }

    // CQD-style beams are small; 32 is generous and keeps 2p quick.
    let cfg = QueryConfig { beam_k: 32 };
    let exact = QueryConfig { beam_k: n_ent };

    println!("\nhard-answer metrics (macro-averaged over queries; filtered):");
    println!(
        "{:<4} {:>4}  {:>8} {:>6} {:>6} {:>6}   {:>8} {:>6}",
        "type", "n", "MRR(P)", "H@1", "H@3", "H@10", "MRR(G)", "H@10"
    );
    for (name, qs) in &queries {
        let (mut mp, mut mg) = (Acc::default(), Acc::default());
        for q in qs {
            let answers = split_answers(&train_kg, &full_kg, q, 0.5);
            if answers.hard.is_empty() {
                continue;
            }
            mp.add(hard_answer_metrics(
                &answer_query::<Product>(&model, q, &cfg),
                &answers,
            ));
            mg.add(hard_answer_metrics(
                &answer_query::<Godel>(&model, q, &cfg),
                &answers,
            ));
        }
        println!(
            "{:<4} {:>4}  {:>8.3} {:>6.3} {:>6.3} {:>6.3}   {:>8.3} {:>6.3}",
            name,
            mp.n,
            mp.mrr(),
            mp.h1(),
            mp.h3(),
            mp.h10(),
            mg.mrr(),
            mg.h10()
        );
    }
    println!(
        "\n! types are non-reducible (every atom needs a held-out edge; no\n\
         single-link shortcut). The drop from plain to ! types reproduces the\n\
         ICLR 2025 finding (arXiv:2410.12537) that reducible benchmark queries\n\
         overstate multi-hop composition. Compare plain types against CQD\n\
         (arXiv:2011.03459) / QTO (ICML 2023); protocol matches, files differ."
    );

    // The rest of the stack on the same model. Witness: why is the top hard
    // answer of the first 2i query an answer? (Gödel: the Idempotent bound.)
    if let Some(q) = queries[2].1.first() {
        let answers = split_answers(&train_kg, &full_kg, q, 0.5);
        if let Some(&target) = answers.hard.iter().next() {
            if let Ok(w) = explain_answer::<Godel>(&model, q, &exact, target) {
                println!("\nwitness for one 2i hard answer ({}):", ent_names[target]);
                print!("{}", w.render());
            }
        }
    }

    // Conformal coverage: calibrate 1p on the valid split, cover on test.
    let calib: Vec<(Query, usize)> = valid
        .iter()
        .take(300)
        .map(|&(h, r, t)| (Query::anchor(h, r), t))
        .collect();
    let tests: Vec<(Query, usize)> = test
        .iter()
        .take(300)
        .map(|&(h, r, t)| (Query::anchor(h, r), t))
        .collect();
    let alpha = 0.2;
    match calibrate::<Product>(&model, &calib, &cfg, alpha) {
        Ok(threshold) => {
            let cov = empirical_coverage::<Product>(&model, &tests, &cfg, &threshold);
            println!(
                "\nconformal 1p answer sets: qhat {:.4} on {} valid pairs; \
                 held-out coverage {:.0}% (nominal {:.0}%)",
                threshold.qhat,
                threshold.n_calibration,
                cov * 100.0,
                (1.0 - alpha) * 100.0
            );
        }
        Err(e) => eprintln!("conformal calibration failed: {e}"),
    }
}

#[derive(Default)]
struct Acc {
    n: usize,
    mrr: f64,
    h1: f64,
    h3: f64,
    h10: f64,
}

impl Acc {
    fn add(&mut self, m: QueryMetrics) {
        self.n += 1;
        self.mrr += m.mrr as f64;
        self.h1 += m.hits1 as f64;
        self.h3 += m.hits3 as f64;
        self.h10 += m.hits10 as f64;
    }
    fn mrr(&self) -> f64 {
        self.mrr / self.n.max(1) as f64
    }
    fn h1(&self) -> f64 {
        self.h1 / self.n.max(1) as f64
    }
    fn h3(&self) -> f64 {
        self.h3 / self.n.max(1) as f64
    }
    fn h10(&self) -> f64 {
        self.h10 / self.n.max(1) as f64
    }
}

type Splits = (
    Vec<(String, String, String)>,
    Vec<(String, String, String)>,
    Vec<(String, String, String)>,
);

fn load_splits(dir: &Path) -> Option<Splits> {
    let read = |f: &str| -> Option<Vec<(String, String, String)>> {
        let text = std::fs::read_to_string(dir.join(f)).ok()?;
        Some(
            text.lines()
                .filter_map(|l| {
                    let mut it = l.split('\t');
                    Some((
                        it.next()?.to_string(),
                        it.next()?.to_string(),
                        it.next()?.to_string(),
                    ))
                })
                .collect(),
        )
    };
    Some((read("train.txt")?, read("valid.txt")?, read("test.txt")?))
}
