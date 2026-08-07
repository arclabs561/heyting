//! The BetaE benchmark query files, evaluated exactly: cross-paper
//! comparability on FB15k-237.
//!
//! Loads the KGReasoning pickles (Ren & Leskovec, NeurIPS 2020;
//! snap-stanford/KGReasoning) — queries, easy/hard answer sets, and the
//! id maps — translates their entity/relation ids into the trained tranz
//! DistMult's vocabulary (their inverse relations `-rel` map to the
//! model's `rel_inv` reciprocal rows), builds a [`Query`] tree per query,
//! and scores hard answers with THEIR easy/hard split, so the numbers are
//! protocol- and file-exact against published tables (BetaE, CQD, QTO)
//! rather than merely protocol-matched like `fb15k237_clqa`'s generated
//! queries.
//!
//! All 14 BetaE query types are tree-form, hence expressible: chains
//! (1p/2p/3p), intersections (2i/3i), projected intersections (pi/ip),
//! unions (2u/up), and the five negation types (2in/3in/inp/pin/pni).
//! EPFO types score under Product and Gödel. Negation types score under
//! Product with the negated branch as a crisp top-k exclusion mask
//! injected via `Query::given` (see the negation note in `main`: soft
//! `1 − sigmoid` negation over uncalibrated degrees is measurably at the
//! random floor at any temperature).
//!
//! Data-gated: needs `data/FB15k-237-betae/` (run
//! `scripts/fetch_betae_fb15k237.sh`, ~1.4 GB download) and the trained
//! DistMult of `fb15k237_clqa` (`data/fb15k237-distmult`). Without them
//! this prints instructions and exits 0.
//!
//! Run: cargo run --release --features tranz --example betae_fb15k237

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use heyting::adapters::PointModel;
use heyting::{
    answer_query, hard_answer_metrics, Godel, Product, Query, QueryAnswers, QueryConfig,
    QueryMetrics,
};
use serde_pickle::{HashableValue, Value};

const DIR: &str = "data/FB15k-237-betae";
const EMB: &str = "data/fb15k237-distmult";
const PER_TYPE: usize = 300;
/// Entities excluded by a negated branch (see the negation note in main).
const NEG_TOP_K: usize = 50;

fn main() {
    let (Some(id2ent), Some(id2rel)) = (
        load_pickle(&format!("{DIR}/id2ent.pkl")),
        load_pickle(&format!("{DIR}/id2rel.pkl")),
    ) else {
        eprintln!("BetaE query files not found: run scripts/fetch_betae_fb15k237.sh");
        return; // data-gated no-op.
    };
    let (Some(queries), Some(easy), Some(hard)) = (
        load_pickle(&format!("{DIR}/test-queries.plain.pkl")),
        load_pickle(&format!("{DIR}/test-easy-answers.plain.pkl")),
        load_pickle(&format!("{DIR}/test-hard-answers.plain.pkl")),
    ) else {
        eprintln!(
            "BetaE plain pickles missing under {DIR}: re-run \
             scripts/fetch_betae_fb15k237.sh (it converts the defaultdict \
             pickles to plain dicts)"
        );
        return;
    };
    let (Some((ent_names, ent_vecs)), Some((rel_names, rel_vecs))) = (
        tranz::io::import_embeddings(&Path::new(EMB).join("entities.tsv")).ok(),
        tranz::io::import_embeddings(&Path::new(EMB).join("relations.tsv")).ok(),
    ) else {
        eprintln!("trained DistMult not found: see examples/fb15k237_clqa.rs header");
        return;
    };

    // Their id -> our embedding row. Inverse relations `-x` -> `x_inv`.
    let ent_row: HashMap<&str, usize> = ent_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let rel_row: HashMap<&str, usize> = rel_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let map_vocab = |v: &Value, f: &dyn Fn(&str) -> Option<usize>| -> HashMap<i64, usize> {
        let Value::Dict(d) = v else {
            return HashMap::new();
        };
        d.iter()
            .filter_map(|(k, name)| {
                let (HashableValue::I64(id), Value::String(name)) = (k, name) else {
                    return None;
                };
                Some((*id, f(name)?))
            })
            .collect()
    };
    let ents = map_vocab(&id2ent, &|n| ent_row.get(n).copied());
    // Their inverse relations `-rel` prefer the model's learned reciprocal
    // `rel_inv` rows (train with --reciprocals). The fallback maps `-rel`
    // onto the forward row, which is EXACT for DistMult's symmetric score
    // but direction-blind: a model without reciprocal rows ranks both
    // argument types of an anti-symmetric relation interleaved, and
    // inverse-direction atoms (half of all atoms in these files) degrade
    // every query type they touch.
    let rels = map_vocab(&id2rel, &|n| match n.split_at(1) {
        ("+", base) => rel_row.get(base).copied(),
        ("-", base) => rel_row
            .get(format!("{base}_inv").as_str())
            .or_else(|| rel_row.get(base))
            .copied(),
        _ => rel_row.get(n).copied(),
    });
    let n_their_ents = dict_len(&id2ent);
    let n_their_rels = dict_len(&id2rel);
    eprintln!(
        "vocab mapped: {}/{} entities, {}/{} relations (unmapped queries are skipped)",
        ents.len(),
        n_their_ents,
        rels.len(),
        n_their_rels,
    );

    let Some(dim) = ent_vecs.first().map(Vec::len).filter(|w| *w > 0) else {
        eprintln!("{EMB}/entities.tsv is empty; retrain (see fb15k237_clqa.rs header)");
        return;
    };
    let distmult = tranz::DistMult::from_vecs(ent_vecs, rel_vecs, dim);
    let model = PointModel::with_temperature(distmult, 5.0);
    let cfg = QueryConfig { beam_k: 32 };

    let Value::Dict(queries) = queries else {
        eprintln!("unexpected shape in test-queries.pkl");
        return;
    };
    let (Value::Dict(easy), Value::Dict(hard)) = (easy, hard) else {
        eprintln!("unexpected shape in answer pickles");
        return;
    };

    // Negation branches evaluate CQD-style: product ⊗ everywhere, with the
    // standard negation 1 − x injected as a Given leaf (the terminal 'n'
    // marker builds the un-negated branch, scores it, and complements).
    // Negation as top-k exclusion. Soft `1 − sigmoid` negation is dead on
    // arrival with uncalibrated embedding degrees, at ANY temperature:
    // hard answers of A ∧ ¬B are type-compatible with B's answers by
    // construction, so the model over-scores them on the negated atom
    // (measured on this checkpoint: a gold ranked 392 of 14541 under B
    // still carried sigmoid 0.993 at temperature 1, and negation MRR sat
    // at the random floor). Published systems recover exactly by
    // calibrating so that only near-top ranks count as "true" (QTO); the
    // transparent version of that is a crisp mask excluding the negated
    // branch's top-k candidates and passing everyone else.
    let std_neg = |sub: Query| -> Option<Query> {
        let d = answer_query::<Product>(&model, &sub, &cfg);
        let mut sorted: Vec<f32> = d.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let theta = sorted.get(NEG_TOP_K).copied().unwrap_or(f32::MAX);
        Some(Query::given(
            d.iter()
                .map(|&x| if x > theta { 0.0 } else { 1.0 })
                .collect(),
        ))
    };
    let mut per_type: HashMap<String, Vec<(Query, QueryAnswers)>> = HashMap::new();
    let mut skipped_unmapped = 0usize;
    let mut unknown: BTreeSet<String> = BTreeSet::new();
    for (structure, qset) in &queries {
        let Some(name) = structure_name(structure) else {
            // The remaining unknowns are the De Morgan re-encodings of the
            // two union types (2u/up as ¬(¬A ∧ ¬B)); the DNF forms above
            // cover the same queries.
            unknown.insert(signature(structure));
            continue;
        };
        let Value::Set(qset) = qset else { continue };
        let bucket = per_type.entry(name.to_string()).or_default();
        for q in qset {
            if bucket.len() >= PER_TYPE {
                break;
            }
            let Some(query) = build_query(q, structure, &ents, &rels, &std_neg) else {
                skipped_unmapped += 1;
                continue;
            };
            let key = q.clone();
            let answers = QueryAnswers {
                easy: answer_set(&easy, &key, &ents),
                hard: answer_set(&hard, &key, &ents),
            };
            if answers.hard.is_empty() {
                continue;
            }
            if std::env::var("BETAE_DEBUG").as_deref() == Ok(name) && bucket.is_empty() {
                let scores = answer_query::<Product>(&model, &query, &cfg);
                let gold = *answers.hard.iter().next().unwrap();
                let rank = scores.iter().filter(|&&s| s > scores[gold]).count() + 1;
                let mut top: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                eprintln!(
                    "DEBUG {name}: instance {q:?}\n  gold {gold} final {} rank {rank}; top-1 {:?}",
                    scores[gold], top[0]
                );
                if let Query::Intersection { branches } = &query {
                    for (i, b) in branches.iter().enumerate() {
                        match b {
                            Query::Given { degrees } => eprintln!(
                                "  branch {i} Given(1-B): gold {} top1 {}",
                                degrees[gold], degrees[top[0].0]
                            ),
                            other => {
                                let d = answer_query::<Product>(&model, other, &cfg);
                                eprintln!(
                                    "  branch {i} {other:?}: gold {} top1 {}",
                                    d[gold], d[top[0].0]
                                );
                            }
                        }
                    }
                }
            }
            bucket.push((query, answers));
        }
    }
    eprintln!("skipped {skipped_unmapped} queries touching unmapped vocab");
    if !unknown.is_empty() {
        eprintln!("unrecognized structures: {unknown:?}");
    }

    println!("\nBetaE test files, hard answers, filtered (n <= {PER_TYPE} per type):");
    println!(
        "{:<5} {:>4}  {:>9} {:>6} {:>6} {:>6}   {:>8} {:>6}",
        "type", "n", "MRR", "H@1", "H@3", "H@10", "MRR(G)", "H@10"
    );
    let table = product_metrics(&per_type, &model, &cfg);
    let godel_by_type: HashMap<&str, Acc> = {
        let mut m = HashMap::new();
        for &name in type_order() {
            let Some(bucket) = per_type.get(name) else {
                continue;
            };
            if name.contains('n') && name != "1p" {
                continue;
            }
            let mut mg = Acc::default();
            for (q, answers) in bucket {
                mg.add(hard_answer_metrics(
                    &answer_query::<Godel>(&model, q, &cfg),
                    answers,
                ));
            }
            m.insert(name, mg);
        }
        m
    };
    for (name, ma) in &table {
        let negation = name.contains('n') && *name != "1p";
        let tag = if negation { "(P\u{00ac})" } else { " (P)" };
        let mg = godel_by_type.get(name).copied().unwrap_or_default();
        println!(
            "{:<5} {:>4}  {:>6.3}{tag} {:>6.3} {:>6.3} {:>6.3}   {:>8.3} {:>6.3}",
            name,
            ma.n,
            ma.mrr(),
            ma.h1(),
            ma.h3(),
            ma.h10(),
            mg.mrr(),
            mg.h10()
        );
    }
    println!(
        "\nSame files as BetaE (NeurIPS 2020) / CQD (ICLR 2021) / QTO (ICML\n\
         2023) FB15k-237 tables: numbers are directly comparable. EPFO rows\n\
         score under Product (P) with Godel (G) alongside; negation rows\n\
         (P¬) exclude the negated branch's top-{NEG_TOP_K} candidates via a\n\
         crisp Given mask (uncalibrated soft negation ranks at the random\n\
         floor). The DistMult here is small (d 256); CQD/QTO tables use\n\
         ComplEx-N3 at d >= 1000, so compare shapes, not absolutes."
    );
}

/// Recognize a BetaE structure tuple and name it. Structures are nested
/// tuples over the markers 'e' (anchor), 'r' (projection), 'n' (negation),
/// 'u' (union).
fn structure_name(s: &HashableValue) -> Option<&'static str> {
    let sig = signature(s);
    Some(match sig.as_str() {
        "(e,(r))" => "1p",
        "(e,(r,r))" => "2p",
        "(e,(r,r,r))" => "3p",
        "((e,(r)),(e,(r)))" => "2i",
        "((e,(r)),(e,(r)),(e,(r)))" => "3i",
        "((e,(r,r)),(e,(r)))" => "pi",
        "(((e,(r)),(e,(r))),(r))" => "ip",
        "((e,(r)),(e,(r)),(u))" => "2u",
        "(((e,(r)),(e,(r)),(u)),(r))" => "up",
        "((e,(r)),(e,(r,n)))" => "2in",
        "((e,(r)),(e,(r)),(e,(r,n)))" => "3in",
        "(((e,(r)),(e,(r,n))),(r))" => "inp",
        "((e,(r,r)),(e,(r,n)))" => "pin",
        "((e,(r,r,n)),(e,(r)))" => "pni",
        _ => return None,
    })
}

/// Canonical text signature of a structure tuple.
fn signature(s: &HashableValue) -> String {
    match s {
        HashableValue::String(m) => m.clone(),
        HashableValue::Tuple(items) => {
            let inner: Vec<String> = items.iter().map(signature).collect();
            format!("({})", inner.join(","))
        }
        _ => "?".to_string(),
    }
}

/// Build a heyting Query from a BetaE query instance, guided by its
/// structure tuple (same nesting; ints where the structure has markers).
/// Returns None when any id is unmapped in the embedding vocabulary.
fn build_query(
    q: &HashableValue,
    s: &HashableValue,
    ents: &HashMap<i64, usize>,
    rels: &HashMap<i64, usize>,
    on_neg: &dyn Fn(Query) -> Option<Query>,
) -> Option<Query> {
    let (HashableValue::Tuple(qs), HashableValue::Tuple(ss)) = (q, s) else {
        return None;
    };
    // Anchor chain or projected sub-query: second element is the marker
    // tuple of 'r'/'n' steps.
    if let Some(HashableValue::Tuple(markers)) = ss.get(1) {
        if markers
            .iter()
            .all(|m| matches!(m, HashableValue::String(x) if x == "r" || x == "n"))
        {
            let HashableValue::Tuple(steps) = qs.get(1)? else {
                return None;
            };
            if markers.len() != steps.len() {
                return None;
            }
            let mut query = match (&qs[0], &ss[0]) {
                (HashableValue::I64(e), HashableValue::String(m)) if m == "e" => {
                    Query::anchor(*ents.get(e)?, rel_of(steps.first()?, rels)?)
                }
                (inner_q, inner_s) => {
                    let sub = build_query(inner_q, inner_s, ents, rels, on_neg)?;
                    sub.then(rel_of(steps.first()?, rels)?)
                }
            };
            // First step consumed above ('n' is never first).
            for (step, marker) in steps.iter().zip(markers.iter()).skip(1) {
                query = match marker {
                    HashableValue::String(m) if m == "r" => query.then(rel_of(step, rels)?),
                    HashableValue::String(m) if m == "n" => on_neg(query)?,
                    _ => return None,
                };
            }
            return Some(query);
        }
    }
    // Intersection / union of branches; a ('u',) marker tags unions.
    let is_union = matches!(
        (ss.last(), qs.last()),
        (Some(HashableValue::Tuple(m)), _) if m.len() == 1
            && matches!(m.first(), Some(HashableValue::String(x)) if x == "u")
    );
    let n = if is_union { ss.len() - 1 } else { ss.len() };
    let branches: Option<Vec<Query>> = (0..n)
        .map(|i| build_query(&qs[i], &ss[i], ents, rels, on_neg))
        .collect();
    let branches = branches?;
    Some(if is_union {
        Query::union(branches)
    } else {
        Query::intersection(branches)
    })
}

/// Resolve one projection step to a mapped relation row.
fn rel_of(step: &HashableValue, rels: &HashMap<i64, usize>) -> Option<usize> {
    match step {
        HashableValue::I64(r) => rels.get(r).copied(),
        _ => None,
    }
}

/// Translate one answer set from their entity ids to embedding rows.
fn answer_set(
    dict: &std::collections::BTreeMap<HashableValue, Value>,
    key: &HashableValue,
    ents: &HashMap<i64, usize>,
) -> BTreeSet<usize> {
    let Some(Value::Set(ids)) = dict.get(key) else {
        return BTreeSet::new();
    };
    ids.iter()
        .filter_map(|v| match v {
            HashableValue::I64(id) => ents.get(id).copied(),
            _ => None,
        })
        .collect()
}

fn dict_len(v: &Value) -> usize {
    match v {
        Value::Dict(d) => d.len(),
        _ => 0,
    }
}

fn load_pickle(path: &str) -> Option<Value> {
    let f = std::fs::File::open(path).ok()?;
    serde_pickle::value_from_reader(std::io::BufReader::new(f), Default::default()).ok()
}

/// Per-type ordering of the printed table (fixed, follows the BetaE paper).
fn type_order() -> &'static [&'static str] {
    &[
        "1p", "2p", "3p", "2i", "3i", "pi", "ip", "2u", "up", "2in", "3in", "inp", "pin", "pni",
    ]
}

/// Macro-averaged Product (P) metrics for every type, in [`type_order`]
/// order. This is the primary column that is directly comparable across
/// papers; both the interactive print and the regression test consume it.
fn product_metrics(
    per_type: &std::collections::HashMap<String, Vec<(Query, QueryAnswers)>>,
    model: &PointModel<tranz::DistMult>,
    cfg: &QueryConfig,
) -> Vec<(&'static str, Acc)> {
    type_order()
        .iter()
        .filter_map(|name| {
            let bucket = per_type.get(*name)?;
            let mut ma = Acc::default();
            for (q, answers) in bucket {
                ma.add(hard_answer_metrics(
                    &answer_query::<Product>(model, q, cfg),
                    answers,
                ));
            }
            Some((*name, ma))
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
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

#[cfg(test)]
mod regression {
    use super::*;

    /// Gold Product (P) metrics captured from a release run with
    /// `data/FB15k-237-betae` and `data/fb15k237-distmult` (PER_TYPE 300).
    /// `(mrr, h10)` per type. Tolerances absorb the 0.001 snapshot wobble while
    /// still catching real protocol or model drift (>0.01).
    ///
    /// Run with: cargo test --features tranz --example betae_fb15k237 -- --ignored
    const GOLD: &[(&str, f64, f64)] = &[
        ("1p", 0.329, 0.500),
        ("2p", 0.025, 0.053),
        ("3p", 0.040, 0.076),
        ("2i", 0.167, 0.289),
        ("3i", 0.180, 0.282),
        ("pi", 0.120, 0.213),
        ("ip", 0.126, 0.230),
        ("2u", 0.071, 0.148),
        ("up", 0.034, 0.059),
        ("2in", 0.010, 0.007),
        ("3in", 0.011, 0.011),
        ("inp", 0.016, 0.035),
        ("pin", 0.009, 0.018),
        ("pni", 0.022, 0.035),
    ];

    /// Load the data-gated inputs and return the computed Product table.
    fn run() -> Option<Vec<(&'static str, Acc)>> {
        let id2ent = load_pickle(&format!("{DIR}/id2ent.pkl"))?;
        let id2rel = load_pickle(&format!("{DIR}/id2rel.pkl"))?;
        let queries = load_pickle(&format!("{DIR}/test-queries.plain.pkl"))?;
        let easy = load_pickle(&format!("{DIR}/test-easy-answers.plain.pkl"))?;
        let hard = load_pickle(&format!("{DIR}/test-hard-answers.plain.pkl"))?;

        let (ent_names, ent_vecs) =
            tranz::io::import_embeddings(&Path::new(EMB).join("entities.tsv")).ok()?;
        let (rel_names, rel_vecs) =
            tranz::io::import_embeddings(&Path::new(EMB).join("relations.tsv")).ok()?;
        let ent_row: HashMap<&str, usize> = ent_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let rel_row: HashMap<&str, usize> = rel_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let map_vocab = |v: &Value, f: &dyn Fn(&str) -> Option<usize>| -> HashMap<i64, usize> {
            let Value::Dict(d) = v else {
                return HashMap::new();
            };
            d.iter()
                .filter_map(|(k, name)| {
                    let (HashableValue::I64(id), Value::String(name)) = (k, name) else {
                        return None;
                    };
                    Some((*id, f(name)?))
                })
                .collect()
        };
        let ents = map_vocab(&id2ent, &|n| ent_row.get(n).copied());
        let rels = map_vocab(&id2rel, &|n| match n.split_at(1) {
            ("+", base) => rel_row.get(base).copied(),
            ("-", base) => rel_row
                .get(format!("{base}_inv").as_str())
                .or_else(|| rel_row.get(base))
                .copied(),
            _ => rel_row.get(n).copied(),
        });
        let dim = ent_vecs.first().map(Vec::len).filter(|w| *w > 0)?;
        let model =
            PointModel::with_temperature(tranz::DistMult::from_vecs(ent_vecs, rel_vecs, dim), 5.0);
        let cfg = QueryConfig { beam_k: 32 };
        let Value::Dict(queries) = queries else {
            return None;
        };
        let (Value::Dict(easy), Value::Dict(hard)) = (easy, hard) else {
            return None;
        };

        let std_neg = |sub: Query| -> Option<Query> {
            let d = answer_query::<Product>(&model, &sub, &cfg);
            let mut sorted: Vec<f32> = d.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let theta = sorted.get(NEG_TOP_K).copied().unwrap_or(f32::MAX);
            Some(Query::given(
                d.iter()
                    .map(|&x| if x > theta { 0.0 } else { 1.0 })
                    .collect(),
            ))
        };

        let mut per_type: HashMap<String, Vec<(Query, QueryAnswers)>> = HashMap::new();
        let mut skipped = 0usize;
        for (structure, qset) in &queries {
            let Some(name) = structure_name(structure) else {
                skipped += 1;
                continue;
            };
            let Value::Set(qset) = qset else { continue };
            let bucket = per_type.entry(name.to_string()).or_default();
            for q in qset {
                if bucket.len() >= PER_TYPE {
                    break;
                }
                let Some(query) = build_query(q, structure, &ents, &rels, &std_neg) else {
                    continue;
                };
                let key = q.clone();
                let answers = QueryAnswers {
                    easy: answer_set(&easy, &key, &ents),
                    hard: answer_set(&hard, &key, &ents),
                };
                if answers.hard.is_empty() {
                    continue;
                }
                bucket.push((query, answers));
            }
        }
        eprintln!("regression: skipped {skipped} unmapped/unrecognized");
        Some(product_metrics(&per_type, &model, &cfg))
    }

    /// Data-gated gold regression. SKIPPED by default; run it explicitly in
    /// RELEASE mode (a debug build re-scoring 14x300 queries over 14.5k
    /// entities is impractically slow):
    ///
    /// ```text
    /// cargo test --release --features tranz --example betae_fb15k237 -- --ignored
    /// ```
    ///
    /// Needs the ~1.4 GB BetaE files + trained DistMult in data/.
    #[test]
    #[ignore]
    fn betae_product_table_matches_gold() {
        let Some(table) = run() else {
            eprintln!("BetaE gold regression: data or embeddings missing; skipping.");
            return;
        };
        let mut found = 0;
        for (gold_name, gold_mrr, gold_h10) in GOLD {
            let Some((_, acc)) = table.iter().find(|(n, _)| n == gold_name) else {
                panic!("type {gold_name} missing from product table");
            };
            found += 1;
            assert!(
                (acc.mrr() - gold_mrr).abs() < 0.01,
                "{gold_name} MRR {:.3} != gold {:.3}",
                acc.mrr(),
                gold_mrr
            );
            assert!(
                (acc.h10() - gold_h10).abs() < 0.01,
                "{gold_name} H@10 {:.3} != gold {:.3}",
                acc.h10(),
                gold_h10
            );
        }
        assert_eq!(found, GOLD.len(), "not all gold types checked");
        eprintln!(
            "BetaE gold regression passed: {} types within tolerance on MRR and H@10.",
            found
        );
    }
}
