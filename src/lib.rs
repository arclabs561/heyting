//! # heyting
//!
//! ## Overview
//!
//! `heyting` answers complex logical queries over a knowledge graph — multi-hop
//! questions with AND, OR, NOT, and implication, like "things that are a mammal
//! and eat fish but are not a cat". It evaluates a query as a computation tree
//! over atomic one-hop lookups, combining the resulting membership degrees with
//! fuzzy-logic connectives. The engine is **geometry-generic**: anything that
//! can answer a one-hop query as `[0, 1]` degrees (point embeddings from
//! `tranz`, region embeddings from `subsume`, or the plain [`FuzzyKg`] here)
//! plugs in through the [`AtomicScorer`] trait. Reach for it when you have
//! trained KG embeddings and want interpretable complex-query answers without
//! training a separate query model.
//!
//! ## The logic is a typed algebra
//!
//! Query degrees live in a *residuated lattice*, and the choice of lattice is a
//! type parameter ([`Truth`]): [`Godel`] (min), [`Product`], [`Lukasiewicz`],
//! or [`Viterbi`] (product with max, the best-derivation semiring).
//! The lattice's residuum `a → b` is the **Heyting implication** the crate is
//! named for, exposed both as [`Truth::residuum`] and as the [`Query::Implication`]
//! connective. Picking the algebra picks the logic with no runtime branch — and
//! the negation each one induces is the load-bearing difference (see [`truth`]):
//! [`Lukasiewicz`] gives the soft involutive `¬a = 1 − a` that ranking wants,
//! while [`Godel`]/[`Product`] give the genuine intuitionistic pseudo-complement.
//!
//! ## Relationship to the ecosystem
//!
//! This is the geometry-generic counterpart of `tranz::query` (CQD-Beam over
//! point embeddings): any point or region model that implements [`AtomicScorer`]
//! gets the same complex-query connectives over it.
//!
//! ## Example
//!
//! ```
//! use heyting::{answer_query_topk, FuzzyKg, Godel, Query, QueryConfig};
//!
//! // 0=animal 1=mammal 2=dog 3=cat; relation 0 = is_a.
//! let mut kg = FuzzyKg::new(4);
//! kg.add_edge(2, 0, 1, 1.0); // dog is_a mammal
//! kg.add_edge(3, 0, 1, 1.0); // cat is_a mammal
//! kg.add_edge(1, 0, 0, 1.0); // mammal is_a animal
//!
//! // (dog is_a ?) AND (cat is_a ?) -> mammal.
//! let q = Query::intersection(vec![Query::anchor(2, 0), Query::anchor(3, 0)]);
//! let top = answer_query_topk::<Godel>(&kg, &q, &QueryConfig::default(), 1);
//! assert_eq!(top[0].0, 1);
//! ```

#![warn(missing_docs)]

pub mod abduce;
pub mod adapters;
pub mod calibration;
pub mod conformal;
pub mod eval;
pub mod kg;
pub mod provenance;
pub mod prune;
pub mod query;
pub mod temporal;
pub mod truth;

pub use abduce::{abduce, AbduceConfig, ConjunctBudget, Hypothesis};
pub use calibration::{AffineSigmoidCalibrator, CalibratedScorer, Calibrator, SoftmaxCalibrator};
pub use conformal::{
    answer_set, answer_set_from_scored_pool, calibrate, empirical_coverage, ConformalError,
    ConformalThreshold,
};
pub use eval::{
    answer_query_report, crisp_answers, hard_answer_metrics, split_answers, QueryAnswerReport,
    QueryAnswers, QueryMetrics,
};
pub use kg::FuzzyKg;
pub use provenance::{explain_answer, Witness, WitnessError};
pub use prune::{answer_query_topk_pruned, CandidateSource};
pub use query::{
    answer_queries, answer_query, answer_query_topk, AtomicScorer, Query, QueryConfig,
    RawProjection, RawScoreOrder,
};
pub use temporal::{TemporalKg, TimeSet, TimeWindow};
pub use truth::{Godel, Lukasiewicz, Product, SelectiveOr, Truth, Viterbi};
