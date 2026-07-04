//! Graded complex-query answering over EL++-style *region* embeddings.
//!
//! Proof of concept for an adapter distinct from [`heyting::adapters::BoxModel`]
//! (which is Query2Box: entities are points, relations are translations). Here
//! entities are **boxes** (concept center + offset, as produced by an EL++
//! ontology trainer like subsume's Box2EL-style `BurnElTrainer`), and the
//! atomic relation is graded **subsumption**: `project(C, SUB)[t]` is the degree
//! to which `C ⊑ t`, i.e. how well concept `C`'s box fits inside concept `t`'s
//! box. heyting then composes those atomic degrees into complex queries
//! (intersection, negation, ...) under a chosen [`Truth`] algebra.
//!
//! What this demonstrates that neither crate does alone:
//! - Box2EL and other EL embedders score only *atomic* `C ⊑ D`.
//! - Query2Box / ConE do complex queries, but over plain KGs with ad-hoc
//!   geometric ops, no EL++ TBox semantics and no residuated-lattice truth.
//!
//! Composing EL++-trained *concept boxes* through heyting's residuated logic
//! answers conjunctive/negated superclass queries that respect the trained
//! ontology, with the t-norm (Godel/Product/Lukasiewicz) as a tunable knob.
//!
//! The ontology below is hand-built so the answers are known: `Dog` and `Cat`
//! sit inside `Mammal` inside `Animal`; `Bird` is in `Animal` but not `Mammal`.
//! `Mammal`'s box is deliberately small, so `Dog`/`Cat` only *partially* fit,
//! making "is `Mammal` a common superclass?" a graded truth whose value depends
//! on the logic.
//!
//! Run: `cargo run --example el_clqa`

use heyting::{answer_query_topk, AtomicScorer, Godel, Lukasiewicz, Product, Query, QueryConfig};

/// An axis-aligned box concept: center and half-width (offset), per dimension.
#[derive(Clone)]
struct Concept {
    name: &'static str,
    center: Vec<f32>,
    offset: Vec<f32>,
}

/// Reserved relation id for graded subsumption `C ⊑ ?`.
const SUB: usize = 0;

/// EL++ concept-box scorer. Entities are boxes; the only relation is graded
/// subsumption. This is the box-entity counterpart of `BoxModel`.
struct ElBoxScorer {
    concepts: Vec<Concept>,
    /// Temperature for the distance-to-degree map `exp(-d / temperature)`.
    temperature: f32,
}

impl ElBoxScorer {
    /// Graded box inclusion `A ⊆ B` as a degree in `[0, 1]`.
    ///
    /// Uses the EL++ inclusion distance `L⊆(A, B) = ‖relu(|cA - cB| + oA - oB)‖`
    /// (0 exactly when every extent of `A` fits within `B`), mapped to a degree
    /// by `exp(-L⊆ / temperature)`: `1` when fully contained, decaying as `A`
    /// pokes out of `B`. This mirrors the objective the trainer minimizes.
    fn subsumption_degree(&self, a: &Concept, b: &Concept) -> f32 {
        let mut acc = 0.0f32;
        for d in 0..a.center.len() {
            let v = ((a.center[d] - b.center[d]).abs() + a.offset[d] - b.offset[d]).max(0.0);
            acc += v * v;
        }
        (-acc.sqrt() / self.temperature).exp()
    }
}

impl AtomicScorer for ElBoxScorer {
    fn num_entities(&self) -> usize {
        self.concepts.len()
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        let n = self.concepts.len();
        if relation != SUB || anchor >= n {
            return vec![0.0; n];
        }
        let a = &self.concepts[anchor];
        self.concepts
            .iter()
            .map(|t| self.subsumption_degree(a, t))
            .collect()
    }
}

fn main() {
    // Concepts as 2D boxes. Mammal is deliberately small so Dog/Cat only
    // partially fit -> "Mammal is a common superclass" is a graded truth.
    let concepts = vec![
        Concept {
            name: "Animal",
            center: vec![0.0, 0.0],
            offset: vec![10.0, 10.0],
        },
        Concept {
            name: "Mammal",
            center: vec![0.0, 0.0],
            offset: vec![2.5, 2.5],
        },
        Concept {
            name: "Dog",
            center: vec![-2.0, 0.0],
            offset: vec![1.0, 1.0],
        },
        Concept {
            name: "Cat",
            center: vec![2.0, 0.0],
            offset: vec![1.0, 1.0],
        },
        Concept {
            name: "Bird",
            center: vec![0.0, 8.0],
            offset: vec![1.0, 1.0],
        },
    ];
    let names: Vec<&str> = concepts.iter().map(|c| c.name).collect();
    let id = |n: &str| names.iter().position(|&x| x == n).unwrap();
    let (animal, mammal, dog, cat, bird) =
        (id("Animal"), id("Mammal"), id("Dog"), id("Cat"), id("Bird"));

    let scorer = ElBoxScorer {
        concepts: concepts.clone(),
        temperature: 1.0,
    };
    let cfg = QueryConfig::default();

    let show = |label: &str, ranked: &[(usize, f32)]| {
        let body: Vec<String> = ranked
            .iter()
            .map(|&(e, d)| format!("{}={:.3}", names[e], d))
            .collect();
        println!("  {label:<12} {}", body.join("  "));
    };

    // --- Query A: common superclasses of Dog and Cat -> {Animal, Mammal} ---
    // Intersection of two atomic subsumptions "Dog ⊑ ?" and "Cat ⊑ ?".
    println!("Query A: X such that Dog ⊑ X AND Cat ⊑ X  (common superclasses)");
    let q_a = Query::intersection(vec![Query::anchor(dog, SUB), Query::anchor(cat, SUB)]);
    let godel = answer_query_topk::<Godel>(&scorer, &q_a, &cfg, 3);
    let product = answer_query_topk::<Product>(&scorer, &q_a, &cfg, 3);
    let luka = answer_query_topk::<Lukasiewicz>(&scorer, &q_a, &cfg, 3);
    show("Godel:", &godel);
    show("Product:", &product);
    show("Lukasiewicz:", &luka);

    // Verify: the top two answers are Animal (certain) and Mammal (partial),
    // in that order, under every algebra -- the known common superclasses.
    for (algo, ranked) in [
        ("Godel", &godel),
        ("Product", &product),
        ("Lukasiewicz", &luka),
    ] {
        let top2: Vec<usize> = ranked.iter().take(2).map(|&(e, _)| e).collect();
        assert_eq!(
            top2,
            vec![animal, mammal],
            "{algo}: top-2 must be Animal, Mammal"
        );
    }
    // Animal fully contains Dog and Cat -> degree 1 in every algebra.
    assert!(
        (godel[0].1 - 1.0).abs() < 1e-3,
        "Animal must be a certain superclass"
    );
    // Mammal is a *graded* common superclass, and the t-norm changes its value:
    // Godel (weakest-link min) > Product > Lukasiewicz (strict). This is the
    // point -- the logic choice sets how accumulated partial-fit is penalized.
    let (g_mammal, p_mammal, l_mammal) = (godel[1].1, product[1].1, luka[1].1);
    assert!(
        g_mammal > p_mammal && p_mammal > l_mammal,
        "t-norm ordering must hold: Godel {g_mammal:.3} > Product {p_mammal:.3} > Lukasiewicz {l_mammal:.3}"
    );
    println!(
        "  -> Mammal graded truth by logic: Godel {g_mammal:.3} > Product {p_mammal:.3} > Lukasiewicz {l_mammal:.3}\n"
    );

    // --- Query B: superclasses of Dog that are NOT superclasses of Bird ---
    // Intersection with a negated branch: distinguishes Mammal (contains Dog,
    // not Bird) from Animal (contains both). Uses Lukasiewicz, whose strong
    // negation is the natural choice for negation-bearing queries.
    println!("Query B: X such that Dog ⊑ X AND NOT (Bird ⊑ X)  (Dog-only superclasses)");
    let q_b = Query::intersection(vec![
        Query::anchor(dog, SUB),
        Query::anchor(bird, SUB).negate(),
    ]);
    let ranked_b = answer_query_topk::<Lukasiewicz>(&scorer, &q_b, &cfg, 4);
    show("Lukasiewicz:", &ranked_b);
    // Exclude the reflexive self-answer (Dog ⊑ Dog is trivially 1), matching the
    // standard EL eval convention of dropping the query entity. Among the proper
    // superclasses, Mammal must rank first and Animal must be suppressed: Animal
    // subsumes Bird, so the negated branch drives its degree down, while Mammal
    // (which does not subsume Bird) survives.
    let proper: Vec<(usize, f32)> = ranked_b
        .iter()
        .copied()
        .filter(|&(e, _)| e != dog)
        .collect();
    assert_eq!(proper[0].0, mammal, "top proper superclass must be Mammal");
    let animal_rank = proper.iter().position(|&(e, _)| e == animal);
    let mammal_deg = proper[0].1;
    let animal_deg = animal_rank.map(|r| proper[r].1).unwrap_or(0.0);
    assert!(
        mammal_deg > animal_deg,
        "Mammal ({mammal_deg:.3}) must outrank Animal ({animal_deg:.3}) under negation"
    );
    println!(
        "  -> excluding reflexive self, Mammal ranks first ({:.3}); Animal is suppressed to {} because it also subsumes Bird\n",
        mammal_deg,
        animal_rank.map_or("outside top-3".into(), |r| format!("{:.3}", proper[r].1))
    );

    println!("All assertions passed: graded EL++ subsumption CLQA over region embeddings.");
    let _ = bird;
}
