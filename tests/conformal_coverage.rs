//! Empirical check of the split-conformal coverage guarantee: calibrate on
//! iid `(query, answer)` pairs, then measure coverage on fresh pairs from the
//! same distribution. The guarantee is marginal and finite-sample, so we
//! assert coverage is at least `1 - alpha - margin` over enough trials that
//! the risk of a false failure is negligible (binomial ~5000 trials).
//!
//! Uses `proptest::test_runner::TestRng` so the run is deterministic.

use heyting::{answer_query, answer_set, calibrate, FuzzyKg, Godel, Query, QueryConfig};
use proptest::test_runner::TestRng;
use rand::Rng;

fn random_kg(rng: &mut impl rand::RngCore, n: usize, nrels: usize, edges: usize) -> FuzzyKg {
    let mut kg = FuzzyKg::new(n);
    for _ in 0..edges {
        kg.add_edge(
            rng.random_range(0..n),
            rng.random_range(0..nrels),
            rng.random_range(0..n),
            rng.random_range(0.05..1.0),
        );
    }
    kg
}

/// Draw `(query, true answer)` where the answer is a real positive-degree
/// tail, so the conformal problem is non-degenerate.
fn draw_pair(
    rng: &mut impl rand::RngCore,
    kg: &FuzzyKg,
    n: usize,
    nrels: usize,
    cfg: &QueryConfig,
) -> (Query, usize) {
    loop {
        let anchor = rng.random_range(0..n);
        let rel = rng.random_range(0..nrels);
        let q = Query::anchor(anchor, rel);
        let deg = answer_query::<Godel>(kg, &q, cfg);
        let tails: Vec<usize> = deg
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > 0.01)
            .map(|(e, _)| e)
            .collect();
        if !tails.is_empty() {
            return (q, tails[rng.random_range(0..tails.len())]);
        }
    }
}

#[test]
fn conformal_coverage_meets_nominal() {
    let mut rng = TestRng::from_seed(proptest::test_runner::RngAlgorithm::ChaCha, &[0x5e; 32]);
    let (n, nrels) = (60, 4);
    let cfg = QueryConfig::default();
    let alpha = 0.2f32;
    let n_cal = 80usize;
    let trials = 5000usize;
    let mut covered = 0usize;

    for _ in 0..trials {
        let kg = random_kg(&mut rng, n, nrels, nrels * n / 2);
        let mut cal = Vec::with_capacity(n_cal);
        for _ in 0..n_cal {
            cal.push(draw_pair(&mut rng, &kg, n, nrels, &cfg));
        }
        let thr = calibrate::<Godel>(&kg, &cal, &cfg, alpha).expect("calibrate");
        let (tq, ta) = draw_pair(&mut rng, &kg, n, nrels, &cfg);
        let set = answer_set::<Godel>(&kg, &tq, &cfg, &thr);
        if set.iter().any(|&(e, _)| e == ta) {
            covered += 1;
        }
    }

    let coverage = covered as f64 / trials as f64;
    let nominal = (1.0 - alpha) as f64;
    // Binomial std ~ sqrt(0.8 * 0.2 / 5000) ~ 0.0057. A margin of 0.02 is
    // ~3.5 sigma, so only a genuine guarantee violation fails, not noise.
    assert!(
        coverage >= nominal - 0.02,
        "coverage {coverage:.4} below nominal {nominal:.4} - margin"
    );
    eprintln!(
        "conformal coverage {coverage:.4} (nominal {nominal:.4}, n_cal {n_cal}, trials {trials})"
    );
}
