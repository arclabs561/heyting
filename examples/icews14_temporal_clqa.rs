//! Temporal complex-query answering on ICEWS14 with a trained TComplEx:
//! timestamp-set hops through the ordinary connectives, end to end.
//!
//! A TComplEx model trained by the tranz CLI supplies time-scoped atom
//! degrees through [`TemporalPointModel`]; a [`TimeSet`] registered against
//! a base relation becomes a virtual relation id, so time-scoped queries
//! are ordinary [`Query`] trees and the whole stack applies: easy/hard
//! metrics, provenance witnesses, conformal answer sets. The exact oracle
//! for the easy/hard split is a [`TemporalKg`] over the same quads with the
//! same virtual ids (point events, day = interval start = end).
//!
//! Query types (deterministic first-N generation from the test split, each
//! anchored at a held-out quad `(h, r, t, τ)`):
//! - `1p`: `(h, r, ?)` unconstrained (existential over the whole axis).
//! - `1p-t`: `(h, r, ?)` within `[τ−3, τ+3]`.
//! - `2i-t`: windowed hop AND a second train-supported windowed in-edge of
//!   the same tail; REDUCIBLE (one held-out atom, arXiv:2410.12537's easy
//!   class).
//! - `2i-t!`: both windowed atoms held out; NON-reducible.
//! - `2u-t`: not-during as a union of the two rays `before(τ−3) ∪
//!   after(τ+3)` — the non-contiguous class. The run also asserts the
//!   equivalent single complement-`TimeSet` hop produces the same degrees
//!   (rays and complement are the same set).
//!
//! Data-gated: needs `data/icews14/{train,valid,test}.txt` and
//! `data/icews14-tcomplex/{entities,relations,times}.tsv`. Run
//! `scripts/fetch_icews14.sh`, then in ../tranz:
//! `cargo run --release --features "burn-ndarray,burn-wgpu" --bin tranz --
//!  train-temporal --data ../heyting/data/icews14 --dim 256 --epochs 100
//!  --batch-size 1024 --lr 0.01 --init-scale 0.01 --label-smoothing 0.1
//!  --n3-reg 0.0025 --time-smooth 1.0
//!  --output ../heyting/data/icews14-tcomplex`
//! (the Lacroix et al. regularizers are worth ~0.18 link-prediction MRR
//! here; keep `--init-scale 0.01` with them, see tranz's changelog).
//! Without them this prints instructions and exits 0.
//!
//! Run: cargo run --release --features tranz --example icews14_temporal_clqa

use std::collections::{HashMap, HashSet};
use std::path::Path;

use heyting::adapters::TemporalPointModel;
use heyting::{
    answer_query, calibrate, empirical_coverage, explain_answer, hard_answer_metrics,
    split_answers, AtomicScorer, Godel, Product, Query, QueryConfig, QueryMetrics, TemporalKg,
    TimeSet, TimeWindow,
};

const DATA: &str = "data/icews14";
const EMB: &str = "data/icews14-tcomplex";
const PER_TYPE: usize = 100;
const ANYTIME_N: usize = 25; // whole-axis hops cost |axis| projections each.
const HALF_WINDOW: usize = 3;

type Quad = (usize, usize, usize, usize);

fn main() {
    let (
        Some(raw),
        Some((ent_names, ent_vecs)),
        Some((rel_names, rel_vecs)),
        Some((times, time_vecs)),
    ) = (
        load_splits(Path::new(DATA)),
        tranz::io::import_embeddings(&Path::new(EMB).join("entities.tsv")).ok(),
        tranz::io::import_embeddings(&Path::new(EMB).join("relations.tsv")).ok(),
        tranz::io::import_embeddings(&Path::new(EMB).join("times.tsv")).ok(),
    )
    else {
        eprintln!("ICEWS14 data or trained embeddings not found.");
        eprintln!("1. scripts/fetch_icews14.sh");
        eprintln!("2. in ../tranz: cargo run --release --features \"burn-ndarray,burn-wgpu\" \\");
        eprintln!("   --bin tranz -- train-temporal --data ../heyting/{DATA} --dim 256 \\");
        eprintln!("   --epochs 100 --batch-size 1024 --lr 0.01 --init-scale 0.01 \\");
        eprintln!("   --label-smoothing 0.1 --n3-reg 0.0025 --time-smooth 1.0 \\");
        eprintln!("   --output ../heyting/{EMB}");
        return; // data-gated no-op.
    };

    // Ids are defined by the embedding TSV row order (the model's vocab).
    let idx = |names: &[String]| -> HashMap<String, usize> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect()
    };
    let (ent_id, rel_id, time_id) = (idx(&ent_names), idx(&rel_names), idx(&times));
    let Some(width) = ent_vecs
        .first()
        .map(Vec::len)
        .filter(|w| w % 2 == 0 && *w > 0)
    else {
        eprintln!("{EMB}/entities.tsv is empty or not complex-valued; retrain (see header)");
        return;
    };
    let dim = width / 2;
    let (n_ent, n_rel, n_time) = (ent_names.len(), rel_names.len(), times.len());
    let model = tranz::temporal::TComplEx::from_vecs(ent_vecs, rel_vecs, time_vecs, dim);
    // Temperature 5, as in the FB15k-237 example: spreads saturated 1-N
    // scores; every ranking metric is invariant.
    let mut model = TemporalPointModel::with_temperature(model, 5.0);

    let to_ids = |split: &[(String, String, String, String)]| -> Vec<Quad> {
        split
            .iter()
            .filter_map(|(h, r, t, s)| {
                Some((
                    *ent_id.get(h)?,
                    *rel_id.get(r)?,
                    *ent_id.get(t)?,
                    *time_id.get(s)?,
                ))
            })
            .collect()
    };
    let train = to_ids(&raw.0);
    let valid = to_ids(&raw.1);
    let test = to_ids(&raw.2);
    eprintln!(
        "ICEWS14: {n_ent} entities, {n_rel} relations, {n_time} timestamps, \
         {}/{}/{} train/valid/test quads, dim {dim}",
        train.len(),
        valid.len(),
        test.len()
    );

    // Exact oracles for the easy/hard split: point events, day as interval.
    let mut train_kg = TemporalKg::new(n_ent, n_rel);
    for &(h, r, t, tau) in &train {
        train_kg.add_fact(h, r, t, tau as f64, tau as f64, 1.0);
    }
    let mut full_kg = train_kg.clone();
    for &(h, r, t, tau) in valid.iter().chain(test.iter()) {
        full_kg.add_fact(h, r, t, tau as f64, tau as f64, 1.0);
    }

    // Register each needed (relation, window) once, on the adapter (TimeSet)
    // and both oracles (TimeWindow), asserting the virtual ids line up.
    let mut windows: HashMap<(usize, usize, usize), usize> = HashMap::new();
    let mut register = |m: &mut TemporalPointModel<tranz::temporal::TComplEx>,
                        tkg: &mut TemporalKg,
                        fkg: &mut TemporalKg,
                        r: usize,
                        lo: usize,
                        hi: usize|
     -> usize {
        *windows.entry((r, lo, hi)).or_insert_with(|| {
            let set = TimeSet::between(lo, hi, n_time);
            let id = m.windowed(r, set).expect("valid registration");
            let w = TimeWindow::Between(lo as f64, hi as f64);
            assert_eq!(tkg.windowed(r, w), Some(id), "oracle id parity");
            assert_eq!(fkg.windowed(r, w), Some(id), "oracle id parity");
            id
        })
    };

    // Ray registrations for the not-during union type.
    let mut rays: HashMap<(usize, usize, usize), (usize, usize)> = HashMap::new();

    // Tail-indexed in-edges for intersection generation.
    let mut in_train: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
    for &(h, r, t, tau) in &train {
        in_train.entry(t).or_default().push((h, r, tau));
    }
    let mut in_heldout: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
    for &(h, r, t, tau) in valid.iter().chain(test.iter()) {
        in_heldout.entry(t).or_default().push((h, r, tau));
    }

    let win = |tau: usize| (tau.saturating_sub(HALF_WINDOW), tau + HALF_WINDOW);
    let mut queries: Vec<(&'static str, Vec<Query>)> = vec![
        ("1p", vec![]),
        ("1p-t", vec![]),
        ("2i-t", vec![]),
        ("2i-t!", vec![]),
        ("2u-t", vec![]),
    ];
    let mut seen: HashSet<(usize, usize, usize, usize)> = HashSet::new();
    for &(h, r, t, tau) in &test {
        if queries[0].1.len() < ANYTIME_N && seen.insert((0, h, r, 0)) {
            queries[0].1.push(Query::anchor(h, r));
        }
        let (lo, hi) = win(tau);
        if queries[1].1.len() < PER_TYPE && seen.insert((1, h, r, tau)) {
            let id = register(&mut model, &mut train_kg, &mut full_kg, r, lo, hi);
            queries[1].1.push(Query::anchor(h, id));
        }
        if queries[2].1.len() < PER_TYPE {
            // Second windowed in-edge of t supported by TRAIN: reducible.
            if let Some(&(h2, r2, tau2)) = in_train
                .get(&t)
                .and_then(|v| v.iter().find(|&&(h2, _, _)| h2 != h))
            {
                if seen.insert((2, h, r, r2)) {
                    let a = register(&mut model, &mut train_kg, &mut full_kg, r, lo, hi);
                    let (lo2, hi2) = win(tau2);
                    let b = register(&mut model, &mut train_kg, &mut full_kg, r2, lo2, hi2);
                    queries[2].1.push(Query::intersection(vec![
                        Query::anchor(h, a),
                        Query::anchor(h2, b),
                    ]));
                }
            }
        }
        if queries[3].1.len() < PER_TYPE {
            // Both windowed atoms held out: non-reducible.
            if let Some(&(h2, r2, tau2)) = in_heldout
                .get(&t)
                .and_then(|v| v.iter().find(|&&(h2, _, _)| h2 != h))
            {
                if seen.insert((3, h, r, r2)) {
                    let a = register(&mut model, &mut train_kg, &mut full_kg, r, lo, hi);
                    let (lo2, hi2) = win(tau2);
                    let b = register(&mut model, &mut train_kg, &mut full_kg, r2, lo2, hi2);
                    queries[3].1.push(Query::intersection(vec![
                        Query::anchor(h, a),
                        Query::anchor(h2, b),
                    ]));
                }
            }
        }
        if queries[4].1.len() < PER_TYPE && seen.insert((4, h, r, tau)) {
            // Not-during [lo, hi] as before(lo) ∪ after(hi).
            let (before_id, after_id) = *rays.entry((r, lo, hi)).or_insert_with(|| {
                let b = if lo > 0 {
                    register(&mut model, &mut train_kg, &mut full_kg, r, 0, lo - 1)
                } else {
                    // Empty ray: an off-axis singleton scores nothing.
                    register(&mut model, &mut train_kg, &mut full_kg, r, n_time, n_time)
                };
                let a = register(
                    &mut model,
                    &mut train_kg,
                    &mut full_kg,
                    r,
                    hi + 1,
                    n_time - 1,
                );
                (b, a)
            });
            queries[4].1.push(Query::union(vec![
                Query::anchor(h, before_id),
                Query::anchor(h, after_id),
            ]));
        }
    }

    // Carrier check: on the adapter, one complement-TimeSet hop must equal
    // the union of the two ray hops (they are the same set of timestamps).
    // On a SCRATCH adapter: probe registrations must not desynchronize the
    // shared virtual-id sequence between the metric adapter and the oracles.
    if let Some(&(h, r, _, tau)) = test.first() {
        let mut probe = TemporalPointModel::with_temperature(model.model.clone(), 5.0);
        let (lo, hi) = win(tau);
        let complement = TimeSet::between(lo, hi, n_time).complement();
        let comp_id = probe.windowed(r, complement).expect("valid");
        // Complement of [lo, hi] = [0, lo-1] ∪ [hi+1, n-1]; an off-axis
        // range registers the empty set.
        let (b_lo, b_hi) = if lo > 0 {
            (0, lo - 1)
        } else {
            (n_time, n_time)
        };
        let before_id = model_windowed_probe(&mut probe, r, b_lo, b_hi, n_time);
        let after_id = model_windowed_probe(&mut probe, r, hi + 1, n_time - 1, n_time);
        let comp = probe.project(h, comp_id);
        let before = probe.project(h, before_id);
        let after = probe.project(h, after_id);
        for ((c, b), a) in comp.iter().zip(&before).zip(&after) {
            assert!(
                (c - b.max(*a)).abs() < 1e-6,
                "complement hop must equal the union of its rays"
            );
        }
        eprintln!("carrier check: complement hop == union of rays (exact)");
    }

    let cfg = QueryConfig { beam_k: 32 };
    let exact = QueryConfig { beam_k: n_ent };

    println!("\nhard-answer metrics (macro-averaged over queries; filtered):");
    println!(
        "{:<6} {:>4}  {:>8} {:>6} {:>6} {:>6}   {:>8} {:>6}",
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
            "{:<6} {:>4}  {:>8.3} {:>6.3} {:>6.3} {:>6.3}   {:>8.3} {:>6.3}",
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
        "\n2i-t! is non-reducible (both windowed atoms need a held-out quad,\n\
         the arXiv:2410.12537 discipline); its query population differs from\n\
         2i-t by construction, so compare within a type across models rather\n\
         than across the two columns. 2u-t hops through the non-contiguous\n\
         not-during set that interval windows cannot carry."
    );

    // Witness: why is the top hard answer of the first 2i-t query an answer?
    if let Some(q) = queries[2].1.first() {
        let answers = split_answers(&train_kg, &full_kg, q, 0.5);
        if let Some(&target) = answers.hard.iter().next() {
            if let Ok(w) = explain_answer::<Godel>(&model, q, &exact, target) {
                println!(
                    "\nwitness for one 2i-t hard answer ({}):",
                    ent_names[target]
                );
                print!("{}", w.render());
            }
        }
    }

    // Conformal coverage on windowed 1p: calibrate on valid, cover on test.
    let mut windowed_pairs = |quads: &[Quad],
                              m: &mut TemporalPointModel<tranz::temporal::TComplEx>,
                              tkg: &mut TemporalKg,
                              fkg: &mut TemporalKg|
     -> Vec<(Query, usize)> {
        quads
            .iter()
            .take(300)
            .map(|&(h, r, t, tau)| {
                let (lo, hi) = win(tau);
                let id = register(m, tkg, fkg, r, lo, hi);
                (Query::anchor(h, id), t)
            })
            .collect()
    };
    let calib = windowed_pairs(&valid, &mut model, &mut train_kg, &mut full_kg);
    let tests = windowed_pairs(&test, &mut model, &mut train_kg, &mut full_kg);
    let alpha = 0.2;
    match calibrate::<Product>(&model, &calib, &cfg, alpha) {
        Ok(threshold) => {
            let cov = empirical_coverage::<Product>(&model, &tests, &cfg, &threshold);
            println!(
                "\nconformal windowed-1p answer sets: qhat {:.4} on {} valid pairs; \
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

/// Register a between-window on the adapter only (probe helper for the
/// carrier check; the metric path registers through the shared oracle map).
fn model_windowed_probe(
    m: &mut TemporalPointModel<tranz::temporal::TComplEx>,
    r: usize,
    lo: usize,
    hi: usize,
    n_time: usize,
) -> usize {
    m.windowed(r, TimeSet::between(lo, hi, n_time))
        .expect("valid probe registration")
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

type RawSplits = (
    Vec<(String, String, String, String)>,
    Vec<(String, String, String, String)>,
    Vec<(String, String, String, String)>,
);

fn load_splits(dir: &Path) -> Option<RawSplits> {
    let read = |f: &str| -> Option<Vec<(String, String, String, String)>> {
        let text = std::fs::read_to_string(dir.join(f)).ok()?;
        Some(
            text.lines()
                .filter_map(|l| {
                    let mut it = l.split('\t');
                    Some((
                        it.next()?.trim().to_string(),
                        it.next()?.trim().to_string(),
                        it.next()?.trim().to_string(),
                        it.next()?.trim().to_string(),
                    ))
                })
                .collect(),
        )
    };
    Some((read("train.txt")?, read("valid.txt")?, read("test.txt")?))
}
