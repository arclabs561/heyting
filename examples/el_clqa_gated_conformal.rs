//! Conformal answer sets over a CONJUNCTIVE readout that is off the atomic seam.
//!
//! `el_clqa_conformal` conformalizes ATOMIC subsumption ("superclasses of C?")
//! through the [`AtomicScorer`](heyting::AtomicScorer) `project(anchor, relation)`
//! seam. The conjunctive least-common-ancestor query "X such that A ⊑ X AND
//! B ⊑ X" cannot use that seam: the winning readout forms the *join* (smallest
//! enclosing box) of A and B geometrically first, then ranks candidates by
//! containment-gated proximity to it (subsume 0.15's [`subsume::clqa::BoxClqa`],
//! which beats both plain containment and a KGE point baseline on GALEN). A
//! containment-min intersection over the atomic seam saturates and fails here.
//!
//! So this example uses the scorer-agnostic conformal core
//! ([`heyting::conformal::calibrate_scores`] +
//! [`heyting::conformal::answer_set_from_degrees`]): compute the gated per-query
//! degree vector yourself, calibrate a threshold on the held-out true-LCA
//! nonconformities, and form coverage-guaranteed LCA sets. The split-conformal
//! guarantee is readout-agnostic, so it transfers to the off-seam readout
//! unchanged: `P(true LCA ∈ set) >= 1 - alpha` for exchangeable queries.
//!
//! The hierarchy is a perfectly nested 1-D interval tree (Thing > 2 supers > 4
//! mids > 8 leaves), so the true LCA of a leaf pair is unambiguous at three
//! depths (same mid, same super, or only the root) and the join-gated readout
//! recovers it. Run:
//! `cargo run --features subsume --example el_clqa_gated_conformal`

use heyting::conformal::{answer_set_from_degrees, calibrate_scores};
use std::collections::HashSet;
use subsume::clqa::BoxClqa;

/// Proximity temperature for the gated readout, scaled to this hierarchy's
/// interval span (~100): large enough that proximity discriminates among the
/// concepts that contain the join, small enough not to wash out.
const TAU: f32 = 10.0;

fn main() {
    // 15 concepts as perfectly nested 1-D intervals (lo, hi). Containment of
    // intervals is subsumption; the join of two intervals is their span.
    let intervals: [(f32, f32); 15] = [
        (0.0, 100.0),  // 0 Thing
        (0.0, 50.0),   // 1 super L
        (50.0, 100.0), // 2 super R
        (0.0, 25.0),   // 3 mid   (under super L)
        (25.0, 50.0),  // 4 mid   (under super L)
        (50.0, 75.0),  // 5 mid   (under super R)
        (75.0, 100.0), // 6 mid   (under super R)
        (0.0, 12.5),   // 7  leaf (under mid 3)
        (12.5, 25.0),  // 8  leaf (under mid 3)
        (25.0, 37.5),  // 9  leaf (under mid 4)
        (37.5, 50.0),  // 10 leaf (under mid 4)
        (50.0, 62.5),  // 11 leaf (under mid 5)
        (62.5, 75.0),  // 12 leaf (under mid 5)
        (75.0, 87.5),  // 13 leaf (under mid 6)
        (87.5, 100.0), // 14 leaf (under mid 6)
    ];
    let n = intervals.len();
    let dim = 1;
    let mut centers = vec![0f32; n];
    let mut offsets = vec![0f32; n];
    for (i, &(lo, hi)) in intervals.iter().enumerate() {
        centers[i] = (lo + hi) / 2.0;
        offsets[i] = (hi - lo) / 2.0;
    }
    let clqa = BoxClqa::new(&centers, &offsets, dim).expect("aligned boxes");

    // Tree structure; the LCA is derived from it, the geometry must recover it.
    let parent = |c: usize| -> Option<usize> {
        match c {
            0 => None,
            1 | 2 => Some(0),
            3 | 4 => Some(1),
            5 | 6 => Some(2),
            7..=14 => Some(3 + (c - 7) / 2),
            _ => None,
        }
    };
    let ancestors = |c: usize| -> Vec<usize> {
        let mut chain = vec![c];
        let mut x = c;
        while let Some(p) = parent(x) {
            chain.push(p);
            x = p;
        }
        chain // [c, parent, ..., root], deepest first
    };
    let lca = |a: usize, b: usize| -> usize {
        let anc_b: HashSet<usize> = ancestors(b).into_iter().collect();
        ancestors(a)
            .into_iter()
            .find(|x| anc_b.contains(x))
            .expect("a shared root exists")
    };

    // The gated LCA degree vector for a query (a, b): join once, then score
    // every concept by containment-gated proximity to that join.
    let degrees_of = |a: usize, b: usize| -> Vec<f32> {
        let (jc, jo) = clqa.join(a, b);
        (0..n).map(|x| clqa.score_lca(&jc, &jo, x, TAU)).collect()
    };

    // Conjunctive queries: every leaf pair, with its true LCA from the tree.
    let leaves: Vec<usize> = (7..15).collect();
    let mut queries: Vec<(usize, usize, usize)> = Vec::new();
    for (i, &a) in leaves.iter().enumerate() {
        for &b in &leaves[i + 1..] {
            queries.push((a, b, lca(a, b)));
        }
    }
    // Deterministic hash-scramble split so calibration and test are exchangeable
    // (an ordered split would put all same-mid pairs in one half and void the
    // guarantee, the gotcha the atomic conformal example documents).
    queries.sort_by_key(|(a, b, _)| {
        a.wrapping_mul(2_654_435_761)
            .wrapping_add(b.wrapping_mul(40_503))
            % 100_000
    });
    let split = queries.len() / 2;
    let (cal, test) = queries.split_at(split);
    println!(
        "{n} concepts, {} conjunctive leaf-pair queries ({} calibration, {} test)",
        queries.len(),
        cal.len(),
        test.len()
    );

    // Confirm the off-seam readout actually recovers the LCA before trusting the
    // conformal numbers: on a nested tree the top-ranked non-query concept
    // should be the true LCA.
    let top1_correct = queries
        .iter()
        .filter(|&&(a, b, l)| {
            let degs = degrees_of(a, b);
            let best = (0..n)
                .filter(|&x| x != a && x != b)
                .max_by(|&x, &y| degs[x].partial_cmp(&degs[y]).unwrap())
                .unwrap();
            best == l
        })
        .count();
    println!(
        "gated readout top-1 LCA accuracy: {}/{}",
        top1_correct,
        queries.len()
    );

    // Nonconformity of each calibration query's true LCA under the gated readout.
    let cal_nonconf: Vec<f32> = cal
        .iter()
        .map(|&(a, b, l)| 1.0 - degrees_of(a, b)[l])
        .collect();

    println!(
        "\n{:<8} {:>8} {:>10} {:>12} {:>12}",
        "alpha", "1-alpha", "q_hat", "empirical", "mean|set|"
    );
    for &alpha in &[0.3f32, 0.2, 0.1] {
        let thr = calibrate_scores(&cal_nonconf, alpha).expect("calibrate");
        let (mut hits, mut set_total) = (0usize, 0usize);
        for &(a, b, l) in test {
            let degs = degrees_of(a, b);
            // Exclude the two query leaves; they cannot be their own ancestor.
            let set: Vec<(usize, f32)> = answer_set_from_degrees(&degs, &thr)
                .into_iter()
                .filter(|&(e, _)| e != a && e != b)
                .collect();
            if set.iter().any(|&(e, _)| e == l) {
                hits += 1;
            }
            set_total += set.len();
        }
        let cov = hits as f32 / test.len() as f32;
        let mean_set = set_total as f32 / test.len() as f32;
        println!(
            "{alpha:<8.2} {:>8.2} {:>10.3} {:>12.3} {:>12.2}",
            1.0 - alpha,
            thr.qhat,
            cov,
            mean_set
        );
        // Finite-sample coverage is >= 1-alpha in expectation; allow modest
        // slack below nominal on this small fixed test split.
        assert!(
            cov >= 1.0 - alpha - 0.25,
            "coverage {cov:.3} far below nominal {:.2} at alpha {alpha}",
            1.0 - alpha
        );
    }

    // One calibrated LCA set: a same-super, different-mid pair whose LCA is a
    // super, at alpha = 0.1.
    let thr = calibrate_scores(&cal_nonconf, 0.1).expect("calibrate");
    let (qa, qb) = (7usize, 9usize); // leaf of mid 3 and leaf of mid 4 -> LCA super L (1)
    let degs = degrees_of(qa, qb);
    let set: Vec<String> = answer_set_from_degrees(&degs, &thr)
        .into_iter()
        .filter(|&(e, _)| e != qa && e != qb)
        .map(|(e, d)| format!("{e}={d:.2}"))
        .collect();
    println!(
        "\nConformal LCA set for (leaf {qa}, leaf {qb}) at alpha=0.1: {}",
        set.join(" ")
    );
    println!("(true LCA: super {} )", lca(qa, qb));

    println!("\nAll assertions passed: conformal LCA sets over the off-seam gated readout.");
}
