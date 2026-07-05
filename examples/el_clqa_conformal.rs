//! Conformal-calibrated CLQA over EL++-style region embeddings.
//!
//! Extends the `el_clqa` proof of concept with a coverage guarantee. Graded
//! subsumption degrees rank superclasses but do not say how many to trust;
//! split conformal prediction (Zhu et al. NAACL 2025, wired in
//! [`heyting::conformal`]) turns them into an answer *set* that contains the
//! true superclass with probability at least `1 - alpha`, calibrated on held-out
//! `(query, true superclass)` pairs.
//!
//! The novelty is the atomic layer: the degrees come from a *faithful EL++
//! geometric model* (concept boxes scored by graded inclusion), not a plain KG
//! link predictor, so the conformal set is over ontology-respecting answers.
//! Conformalized answer sets have been done for KGE (2408.08248, 2505.16877)
//! but not over a faithful EL model. That atomic layer is the reusable
//! [`heyting::adapters::FaithfulBoxModel`] adapter (entities are boxes,
//! containment is subsumption); this example is its worked conformal use.
//!
//! The hierarchy has a deliberately tight root, so mid concepts only partially
//! fit inside it: the true subsumption degrees are graded (mid ⊑ Thing ~0.37,
//! leaf ⊑ Thing ~0.82, leaf ⊑ mid ~1.0), which is what makes the conformal
//! threshold non-trivial. Run: `cargo run --example el_clqa_conformal`

use heyting::adapters::FaithfulBoxModel;
use heyting::{AtomicScorer, Godel, Query, QueryConfig};

/// Reserved relation id for graded subsumption `C ⊑ ?`.
const SUB: usize = 0;

fn main() {
    // 13 concepts: Thing (root, tight box) > 4 mids > 8 leaves (2 per mid).
    // Boxes are laid out so containment is real but partial (graded degrees).
    let mut centers: Vec<Vec<f32>> = vec![vec![0.0, 0.0]]; // Thing
    let mut offsets: Vec<Vec<f32>> = vec![vec![6.5, 6.5]];
    let mid_pos = [[4.0f32, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0]];
    for p in mid_pos {
        centers.push(vec![p[0], p[1]]);
        offsets.push(vec![3.5, 3.5]);
    }
    // Two leaves per mid, offset along the axis away from / toward the root.
    for (m, p) in mid_pos.iter().enumerate() {
        for s in [1.5f32, -1.5] {
            let axis = m % 2; // mids 0,2 spread on x; 1,3 on y
            let mut c = *p;
            c[axis] += s;
            centers.push(vec![c[0], c[1]]);
            offsets.push(vec![1.2, 1.2]);
        }
    }
    let n = centers.len();
    // The faithful EL++ box adapter: subsumption on relation SUB, graded
    // inclusion degrees. Same geometry the inline scorer used before, now the
    // reusable library type.
    let scorer = FaithfulBoxModel::new(centers, offsets)
        .expect("aligned box dimensions")
        .with_subsumption(SUB);

    // Parent of each concept (None for root). Mids -> Thing; leaves -> their mid.
    let parent = |c: usize| -> Option<usize> {
        match c {
            0 => None,
            1..=4 => Some(0),
            _ => Some(1 + (c - 5) / 2),
        }
    };
    // (query, true superclass) pairs from the transitive closure: for each
    // concept, every proper ancestor is a true answer to "superclass of C?".
    let mut raw: Vec<(usize, usize)> = Vec::new();
    for c in 0..n {
        let mut a = parent(c);
        while let Some(anc) = a {
            raw.push((c, anc));
            a = parent(anc);
        }
    }
    // Deterministic pseudo-random split (a hash scramble) so calibration and
    // test are exchangeable. Conformal's coverage guarantee needs exchangeability;
    // an ordered split (e.g. by depth) puts all the easy degree-1 pairs in one
    // half and the graded pairs in the other, voiding the guarantee.
    raw.sort_by_key(|(c, anc)| {
        c.wrapping_mul(2_654_435_761)
            .wrapping_add(anc.wrapping_mul(40_503))
            % 100_000
    });
    let split = raw.len() / 2;
    let cal: Vec<(Query, usize)> = raw[..split]
        .iter()
        .map(|&(c, a)| (Query::anchor(c, SUB), a))
        .collect();
    let test: Vec<(Query, usize)> = raw[split..]
        .iter()
        .map(|&(c, a)| (Query::anchor(c, SUB), a))
        .collect();
    println!(
        "{} concepts, {} subsumption pairs ({} calibration, {} test)",
        n,
        raw.len(),
        cal.len(),
        test.len()
    );

    let cfg = QueryConfig::default();
    // Show the graded degrees the conformal layer works over.
    let leaf = 5; // first leaf, under mid 1
    let leaf_degrees = scorer.project(leaf, SUB);
    println!(
        "\nSample true-subsumption degrees for leaf {leaf}: ⊑mid={:.3}  ⊑Thing={:.3}",
        leaf_degrees[1], leaf_degrees[0]
    );

    // Calibrate at several confidence levels and check empirical coverage.
    println!(
        "\n{:<8} {:>8} {:>10} {:>12}",
        "alpha", "1-alpha", "q_hat", "empirical"
    );
    for &alpha in &[0.3f32, 0.2, 0.1] {
        let thr =
            heyting::conformal::calibrate::<Godel>(&scorer, &cal, &cfg, alpha).expect("calibrate");
        let cov = heyting::conformal::empirical_coverage::<Godel>(&scorer, &test, &cfg, &thr);
        println!(
            "{alpha:<8.2} {:>8.2} {:>10.3} {:>12.3}",
            1.0 - alpha,
            thr.qhat,
            cov
        );
        // Finite-sample conformal coverage is >= 1-alpha in expectation; on a
        // small fixed test split allow modest slack below the nominal level.
        assert!(
            cov >= 1.0 - alpha - 0.2,
            "coverage {cov:.3} far below nominal {:.2} at alpha {alpha}",
            1.0 - alpha
        );
    }

    // Show one calibrated answer set: the coverage-guaranteed superclasses of a
    // leaf at alpha=0.1, best first.
    let thr = heyting::conformal::calibrate::<Godel>(&scorer, &cal, &cfg, 0.1).expect("calibrate");
    let set =
        heyting::conformal::answer_set::<Godel>(&scorer, &Query::anchor(leaf, SUB), &cfg, &thr);
    let body: Vec<String> = set.iter().map(|(e, d)| format!("{e}={d:.2}")).collect();
    println!(
        "\nConformal superclass set for leaf {leaf} (alpha=0.1): {}",
        body.join(" ")
    );
    println!("(true ancestors: mid 1 and Thing 0)");

    println!("\nAll assertions passed: conformal answer sets over a faithful EL++ model.");
}
