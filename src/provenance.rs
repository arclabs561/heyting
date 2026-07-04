//! Why-provenance witnesses: the proof behind a degree-path answer.
//!
//! Provenance semirings (Green, Karvounarakis & Tannen, PODS 2007) annotate
//! every tuple with an element of a commutative semiring and show that
//! positive query evaluation factors through *provenance polynomials*: the
//! output annotation documents which inputs contributed and how. The Gödel
//! algebra is exactly that paper's *fuzzy semiring* `([0, 1], max, min)`,
//! and because both of its operations are idempotent the provenance
//! polynomial collapses to a single **bottleneck witness tree**: the
//! derivation whose weakest link is strongest. [`explain_answer`] extracts
//! that tree for one `(query, entity)` pair — which atomic facts carried the
//! answer, through which intermediates — with the invariant that the
//! witness's degree equals the engine's [`answer_query`] degree exactly.
//!
//! Scope is deliberate: witnesses require the [`SelectiveOr`] bound (Gödel, Viterbi). `Product` and
//! `Lukasiewicz` are not semirings at all (`⊗` fails to distribute over
//! their `⊕`; e.g. under Łukasiewicz at `a = b = c = 0.5`,
//! `a ⊗ (b ⊕ c) = 0.5` but `(a ⊗ b) ⊕ (a ⊗ c) = 0`), so a single-derivation
//! witness would misreport their degrees, which genuinely aggregate across
//! derivations. Negation and implication are covered by *refutation* and
//! *residuum* markers: the witness of `¬q` is the recorded absence of any
//! derivation of `q` (Grädel & Tannen's dual indeterminates, arXiv:1712.01980,
//! in witness form — negated literals are tracked outside `⊕`/`⊗`), and the
//! witness of `p → c` records the premise degree with the conclusion's
//! witness when one exists. Recursion/fixpoints (path queries) would need the
//! absorptive-polynomial carrier (Dannert-Grädel-Naaf-Tannen, CSL 2021); the
//! tree-form `Query` has none, so witnesses stay polynomial-free.
//!
//! This is the degree-path counterpart of the geometric
//! `materialize_explained` (feature `subsume`): one explains via regions,
//! this one via the facts themselves.
//!
//! [`answer_query`]: crate::answer_query

use crate::query::{answer_query, AtomicScorer, Query, QueryConfig};
use crate::truth::SelectiveOr;

/// A bottleneck derivation tree for one answer entity.
#[derive(Debug, Clone, PartialEq)]
pub enum Witness {
    /// An atomic fact `(anchor, relation, entity)` carrying `degree`.
    Fact {
        /// Hop anchor.
        anchor: usize,
        /// Hop relation.
        relation: usize,
        /// The answer entity this fact supports.
        entity: usize,
        /// The fact's membership degree.
        degree: f32,
    },
    /// A [`Query::Given`] leaf's precomputed degree for the entity.
    Given {
        /// The answer entity.
        entity: usize,
        /// The leaf's degree.
        degree: f32,
    },
    /// A conjunction: every branch's witness, degree = the minimum.
    All {
        /// One witness per branch, in query order.
        branches: Vec<Witness>,
        /// The bottleneck (minimum) degree.
        degree: f32,
    },
    /// A disjunction: the winning branch only.
    Any {
        /// Index of the winning branch in the query's branch list.
        branch: usize,
        /// The winning branch's witness.
        inner: Box<Witness>,
        /// Its degree.
        degree: f32,
    },
    /// A refutation: the recorded absence of any derivation under the
    /// negated sub-query (`degree = ¬inner_degree`, crisp under
    /// [`SelectiveOr`] algebras).
    Refutation {
        /// The answer entity.
        entity: usize,
        /// The negated sub-query's degree at the entity.
        inner_degree: f32,
        /// `T::neg(inner_degree)`.
        degree: f32,
    },
    /// A residuum `p → c`: satisfied vacuously when `p ≤ c`, otherwise
    /// carried by the conclusion.
    Implied {
        /// The premise's degree at the entity.
        premise_degree: f32,
        /// The conclusion's witness, when the conclusion has any support.
        conclusion: Option<Box<Witness>>,
        /// `T::residuum(premise, conclusion)`.
        degree: f32,
    },
    /// An existential hop: the winning intermediate and its two legs.
    Via {
        /// The intermediate entity the existential chose.
        intermediate: usize,
        /// Witness for the intermediate under the inner query.
        inner: Box<Witness>,
        /// The final hop fact from the intermediate to the answer.
        hop: Box<Witness>,
        /// `min(inner, hop)`, the path's bottleneck.
        degree: f32,
    },
}

impl Witness {
    /// The witness's degree; equals the engine's Gödel degree by
    /// construction (property-tested).
    pub fn degree(&self) -> f32 {
        match self {
            Witness::Fact { degree, .. }
            | Witness::Given { degree, .. }
            | Witness::All { degree, .. }
            | Witness::Any { degree, .. }
            | Witness::Refutation { degree, .. }
            | Witness::Implied { degree, .. }
            | Witness::Via { degree, .. } => *degree,
        }
    }

    /// Indented one-line-per-node rendering.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.render_into(&mut out, 0);
        out
    }

    fn render_into(&self, out: &mut String, depth: usize) {
        use std::fmt::Write;
        let pad = "  ".repeat(depth);
        match self {
            Witness::Fact {
                anchor,
                relation,
                entity,
                degree,
            } => {
                let _ = writeln!(
                    out,
                    "{pad}fact ({anchor}, r{relation}, {entity}) [{degree:.3}]"
                );
            }
            Witness::Given { entity, degree } => {
                let _ = writeln!(out, "{pad}given({entity}) [{degree:.3}]");
            }
            Witness::All { branches, degree } => {
                let _ = writeln!(out, "{pad}all [{degree:.3}]");
                for b in branches {
                    b.render_into(out, depth + 1);
                }
            }
            Witness::Any {
                branch,
                inner,
                degree,
            } => {
                let _ = writeln!(out, "{pad}any: branch {branch} [{degree:.3}]");
                inner.render_into(out, depth + 1);
            }
            Witness::Refutation {
                entity,
                inner_degree,
                degree,
            } => {
                let _ = writeln!(
                    out,
                    "{pad}refuted({entity}): inner degree {inner_degree:.3} [{degree:.3}]"
                );
            }
            Witness::Implied {
                premise_degree,
                conclusion,
                degree,
            } => {
                let _ = writeln!(
                    out,
                    "{pad}implied: premise {premise_degree:.3} [{degree:.3}]"
                );
                if let Some(c) = conclusion {
                    c.render_into(out, depth + 1);
                }
            }
            Witness::Via {
                intermediate,
                inner,
                hop,
                degree,
            } => {
                let _ = writeln!(out, "{pad}via {intermediate} [{degree:.3}]");
                inner.render_into(out, depth + 1);
                hop.render_into(out, depth + 1);
            }
        }
    }
}

/// Why [`explain_answer`] can refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessError {
    /// The entity's degree is zero: nothing derives it, so no witness exists.
    ZeroDegree,
}

impl std::fmt::Display for WitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDegree => write!(f, "degree is zero: no derivation supports this entity"),
        }
    }
}

impl std::error::Error for WitnessError {}

/// The bottleneck witness for `entity` under `query`.
///
/// Bounded by [`SelectiveOr`], which is what makes a single-derivation
/// witness exact: a selective `⊕` is `max`, so every degree is realized by
/// one best derivation (bottleneck under Gödel, best product under Viterbi). Evaluates sub-queries
/// with the ordinary dense engine and reads the argmax/argmin chain off the
/// degree vectors, so the returned tree's [`Witness::degree`] equals
/// `answer_query::<T>(..)[entity]` exactly.
pub fn explain_answer<T: SelectiveOr>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
    entity: usize,
) -> Result<Witness, WitnessError> {
    let w = witness::<T>(scorer, query, config, entity)?;
    if w.degree() <= 0.0 {
        return Err(WitnessError::ZeroDegree);
    }
    Ok(w)
}

fn witness<T: SelectiveOr>(
    scorer: &dyn AtomicScorer,
    query: &Query,
    config: &QueryConfig,
    entity: usize,
) -> Result<Witness, WitnessError> {
    match query {
        Query::Anchor {
            entity: anchor,
            relation,
        } => {
            let degree = scorer
                .project_subset(*anchor, *relation, &[entity])
                .first()
                .copied()
                .unwrap_or(0.0);
            Ok(Witness::Fact {
                anchor: *anchor,
                relation: *relation,
                entity,
                degree,
            })
        }
        Query::Given { degrees } => Ok(Witness::Given {
            entity,
            degree: degrees.get(entity).copied().unwrap_or(0.0),
        }),
        Query::Intersection { branches } => {
            let ws: Vec<Witness> = branches
                .iter()
                .map(|b| witness::<T>(scorer, b, config, entity))
                .collect::<Result<_, _>>()?;
            // Idempotent ⟹ ⊗ = min: the conjunction's degree is the bottleneck.
            let degree = ws.iter().map(Witness::degree).fold(T::top(), T::and);
            Ok(Witness::All {
                branches: ws,
                degree,
            })
        }
        Query::Union { branches } => {
            // Gödel ⊕ is max and idempotent: one disjunct carries the answer.
            let ws: Vec<Witness> = branches
                .iter()
                .map(|b| witness::<T>(scorer, b, config, entity))
                .collect::<Result<_, _>>()?;
            let (branch, best) = ws
                .into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.degree()
                        .partial_cmp(&b.degree())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or(WitnessError::ZeroDegree)?;
            let degree = best.degree();
            Ok(Witness::Any {
                branch,
                inner: Box::new(best),
                degree,
            })
        }
        Query::Project { inner, relation } => {
            // The engine expands the top-beam_k intermediates; the witness
            // reads the winning one off the same evaluation, so it agrees
            // with the engine even when the beam truncates.
            let inner_scores = answer_query::<T>(scorer, inner, config);
            let mut order: Vec<usize> = (0..inner_scores.len()).collect();
            order.sort_unstable_by(|&a, &b| {
                inner_scores[b]
                    .partial_cmp(&inner_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut best: Option<(usize, f32, f32)> = None; // (v, path degree, hop degree)
            for &v in order.iter().take(config.beam_k) {
                let iv = inner_scores[v];
                if iv <= 0.0 {
                    break;
                }
                let hop = scorer
                    .project_subset(v, *relation, &[entity])
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let path = T::and(iv, hop);
                if best.is_none_or(|(_, bp, _)| path > bp) {
                    best = Some((v, path, hop));
                }
            }
            let (v, path, hop_degree) = best.ok_or(WitnessError::ZeroDegree)?;
            let inner_witness = witness::<T>(scorer, inner, config, v)?;
            Ok(Witness::Via {
                intermediate: v,
                inner: Box::new(inner_witness),
                hop: Box::new(Witness::Fact {
                    anchor: v,
                    relation: *relation,
                    entity,
                    degree: hop_degree,
                }),
                degree: path,
            })
        }
        Query::Negation { inner } => {
            let inner_degree = answer_query::<T>(scorer, inner, config)
                .get(entity)
                .copied()
                .unwrap_or(0.0);
            Ok(Witness::Refutation {
                entity,
                inner_degree,
                degree: T::neg(inner_degree),
            })
        }
        Query::Implication {
            premise,
            conclusion,
        } => {
            let p = answer_query::<T>(scorer, premise, config)
                .get(entity)
                .copied()
                .unwrap_or(0.0);
            let conclusion_witness = witness::<T>(scorer, conclusion, config, entity)
                .ok()
                .filter(|w| w.degree() > 0.0);
            let c = conclusion_witness.as_ref().map_or(0.0, Witness::degree);
            Ok(Witness::Implied {
                premise_degree: p,
                conclusion: conclusion_witness.map(Box::new),
                degree: T::residuum(p, c),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kg::FuzzyKg;
    use crate::truth::{Godel, Viterbi};

    /// 0=animal 1=mammal 2=dog 3=cat 4=fish; 0=is_a, 1=eats. Two paths from
    /// dog to animal exist only via mammal; degrees are all distinct so
    /// bottlenecks are unambiguous.
    fn kg() -> FuzzyKg {
        let mut kg = FuzzyKg::new(5);
        kg.add_edge(2, 0, 1, 0.9); // dog is_a mammal
        kg.add_edge(3, 0, 1, 0.8); // cat is_a mammal
        kg.add_edge(1, 0, 0, 0.7); // mammal is_a animal
        kg.add_edge(2, 1, 4, 0.4); // dog eats fish
        kg
    }

    /// The soundness invariant: witness degree == dense engine degree, across
    /// every supported shape, under BOTH SelectiveOr algebras (bottleneck
    /// witnesses for Gödel, best-product witnesses for Viterbi).
    #[test]
    fn witness_degree_matches_engine_degree() {
        fn check<T: crate::truth::SelectiveOr>(kg: &FuzzyKg, cfg: &QueryConfig, label: &str) {
            let queries = [
                Query::anchor(2, 0),
                Query::anchor(2, 0).then(0),
                Query::intersection(vec![Query::anchor(2, 0), Query::anchor(3, 0)]),
                Query::union(vec![Query::anchor(2, 0), Query::anchor(2, 1)]),
                Query::intersection(vec![
                    Query::anchor(2, 0).then(0),
                    Query::given(vec![0.6; 5]),
                ]),
                Query::anchor(2, 0).negate(),
                Query::anchor(2, 0).implies(Query::anchor(3, 0)),
            ];
            for q in &queries {
                let dense = answer_query::<T>(kg, q, cfg);
                for (e, &d) in dense.iter().enumerate() {
                    if d > 0.0 {
                        let w =
                            explain_answer::<T>(kg, q, cfg, e).expect("witness for nonzero degree");
                        assert!(
                            (w.degree() - d).abs() < 1e-6,
                            "{label} entity {e}: witness {} vs engine {d}\n{}",
                            w.degree(),
                            w.render()
                        );
                    }
                }
            }
        }
        let kg = kg();
        let cfg = QueryConfig::default();
        check::<Godel>(&kg, &cfg, "godel");
        check::<Viterbi>(&kg, &cfg, "viterbi");
    }

    /// The 2p witness names the true bottleneck path: dog -is_a-> mammal
    /// (0.9) -is_a-> animal (0.7), bottleneck 0.7 at the second hop.
    #[test]
    fn chain_witness_names_the_intermediate() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let q = Query::anchor(2, 0).then(0);
        let w = explain_answer::<Godel>(&kg, &q, &cfg, 0).unwrap();
        match &w {
            Witness::Via {
                intermediate,
                degree,
                ..
            } => {
                assert_eq!(*intermediate, 1, "must route via mammal");
                assert!((degree - 0.7).abs() < 1e-6);
            }
            other => panic!("expected Via, got {other:?}"),
        }
        assert!(w.render().contains("via 1"), "{}", w.render());
    }

    /// Union picks the strongest disjunct and says which.
    #[test]
    fn union_witness_names_the_winning_branch() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let q = Query::union(vec![Query::anchor(2, 1), Query::anchor(2, 0)]);
        // Entity 1 (mammal) is only reachable through branch 1 (is_a).
        let w = explain_answer::<Godel>(&kg, &q, &cfg, 1).unwrap();
        match w {
            Witness::Any { branch, degree, .. } => {
                assert_eq!(branch, 1);
                assert!((degree - 0.9).abs() < 1e-6);
            }
            other => panic!("expected Any, got {other:?}"),
        }
    }

    #[test]
    fn zero_degree_refuses_and_negation_witnesses_are_refutations() {
        let kg = kg();
        let cfg = QueryConfig::default();
        assert_eq!(
            explain_answer::<Godel>(&kg, &Query::anchor(2, 0), &cfg, 4).unwrap_err(),
            WitnessError::ZeroDegree
        );
        // dog is_a fish has degree 0, so its negation at fish is a
        // full-degree refutation (crisp Gödel negation).
        let w = explain_answer::<Godel>(&kg, &Query::anchor(2, 0).negate(), &cfg, 4).unwrap();
        match w {
            Witness::Refutation {
                inner_degree,
                degree,
                ..
            } => {
                assert_eq!(inner_degree, 0.0);
                assert_eq!(degree, 1.0);
            }
            other => panic!("expected Refutation, got {other:?}"),
        }
        // A supported answer's negation is degree 0: no witness.
        assert_eq!(
            explain_answer::<Godel>(&kg, &Query::anchor(2, 0).negate(), &cfg, 1).unwrap_err(),
            WitnessError::ZeroDegree
        );
    }
}
