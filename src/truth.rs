//! Truth-degree algebras: the residuated lattices that logical query answering
//! composes in.
//!
//! A complex query is a formula in many-valued logic, and its answer degrees
//! live in `[0, 1]`. The algebra of those degrees is a *residuated lattice*: a
//! conjunction `⊗` (a t-norm), its lattice dual disjunction `⊕` (a t-conorm),
//! and the **residuum** `→` (the namesake Heyting implication) defined by the
//! adjunction
//!
//! ```text
//! a ⊗ c ≤ b   ⟺   c ≤ (a → b)
//! ```
//!
//! Negation is the pseudo-complement `¬a = a → ⊥`. This is the structure named
//! by [`Heyting`](https://en.wikipedia.org/wiki/Heyting_algebra): the Gödel
//! algebra below *is* a Heyting algebra, and `→` is its implication.
//!
//! # The three canonical algebras
//!
//! Every continuous t-norm is an ordinal sum of these three (Mostert–Shields),
//! so they are the natural basis to expose:
//!
//! | Algebra | `a ⊗ b` | `a → b` | `¬a = a → ⊥` |
//! |---|---|---|---|
//! | [`Godel`] | `min(a, b)` | `1 if a ≤ b else b` | `1 if a = 0 else 0` (crisp) |
//! | [`Product`] | `a · b` | `1 if a ≤ b else b / a` | `1 if a = 0 else 0` (crisp) |
//! | [`Lukasiewicz`] | `max(0, a + b − 1)` | `min(1, 1 − a + b)` | `1 − a` (involutive) |
//!
//! # Which to use
//!
//! The negation column is the load-bearing distinction. [`Godel`] and
//! [`Product`] give the genuine *intuitionistic* pseudo-complement, which is
//! crisp (anything with nonzero degree negates to 0) and therefore useless for
//! *ranking* negated answers. [`Lukasiewicz`] is the involutive MV-algebra whose
//! negation is `1 − a`, matching the soft complement used by BetaE and
//! `tranz::query`. So: reach for [`Lukasiewicz`] on negation-bearing queries,
//! [`Godel`] (min) for intersections, [`Product`] for chains (it propagates
//! magnitude through hops). That region negation is only a pseudo-complement,
//! not a Boolean one, is exactly why geometric CLQA negation is approximate.
//!
//! # Which algebras are semirings
//!
//! Provenance semirings (Green, Karvounarakis & Tannen, PODS 2007) require
//! `(⊕, ⊗)` to form a commutative semiring — in particular `⊗` must
//! distribute over `⊕` — and in exchange give the factorization theorem
//! (evaluation factors through provenance polynomials) and homomorphism
//! commutation. Only [`Godel`] qualifies: `(min, max)` on `[0, 1]` is that
//! paper's *fuzzy semiring*, a distributive lattice. The other two fail
//! distributivity outright: under [`Product`],
//! `a ⊗ (b ⊕ c) = ab + ac − abc` but `(a⊗b) ⊕ (a⊗c) = ab + ac − a²bc`;
//! under [`Lukasiewicz`] at `a = b = c = 0.5`, `a ⊗ (b ⊕ c) = 0.5` while
//! `(a⊗b) ⊕ (a⊗c) = 0`. They are residuated lattices, not semirings — so
//! degree under them genuinely *aggregates across derivations* (the same
//! fact reached through two branches raises the answer), which is
//! how-provenance counting leaking into the degree, not a bug. The
//! [`SelectiveOr`] marker encodes the semiring side of this line;
//! [`crate::provenance`]'s soundness is bounded by it. [`Idempotent`]
//! additionally marks the both-ops-idempotent case (Gödel alone).

/// A residuated lattice of truth degrees on `[0, 1]`.
///
/// The algebra a [`crate::Query`] is evaluated in. Implementors are zero-sized
/// marker types ([`Godel`], [`Product`], [`Lukasiewicz`]) selected as a type
/// parameter, so the choice of logic is made at the call site with no runtime
/// branch.
///
/// # Laws
///
/// Implementors must satisfy, for all `a, b, c ∈ [0, 1]`:
///
/// - `⊗` is associative, commutative, monotone, with unit [`top`](Truth::top):
///   `and(a, 1) = a`.
/// - **Residuation** (the defining adjunction): `and(a, c) ≤ b ⟺ c ≤ residuum(a, b)`.
///   Equivalently the two inequalities `and(a, residuum(a, b)) ≤ b` (modus
///   ponens) and `c ≤ residuum(a, and(a, c))`.
/// - `residuum(a, b) = 1` whenever `a ≤ b` (reflexivity of the order).
///
/// The provided [`Truth::neg`] derives negation from the residuum, so a correct
/// `residuum` yields a correct `¬`.
pub trait Truth {
    /// `⊤`, the top (fully true) degree. `1.0`.
    #[inline]
    fn top() -> f32 {
        1.0
    }

    /// `⊥`, the bottom (false) degree. `0.0`.
    #[inline]
    fn bot() -> f32 {
        0.0
    }

    /// Conjunction `a ⊗ b` (the t-norm). Used for AND and for combining a hop's
    /// premise with its projected scores.
    fn and(a: f32, b: f32) -> f32;

    /// Disjunction `a ⊕ b` (the t-conorm), the algebra's standard dual of `⊗`.
    /// Used for OR.
    fn or(a: f32, b: f32) -> f32;

    /// The residuum `a → b = sup { c : a ⊗ c ≤ b }` — the Heyting implication.
    ///
    /// This is the operation that makes the lattice *residuated*; [`Truth::neg`]
    /// is defined from it.
    fn residuum(a: f32, b: f32) -> f32;

    /// Negation, the pseudo-complement `¬a = a → ⊥`.
    ///
    /// Crisp for [`Godel`]/[`Product`], involutive (`1 − a`) for
    /// [`Lukasiewicz`]. Override only to pin a non-canonical negation.
    #[inline]
    fn neg(a: f32) -> f32 {
        Self::residuum(a, Self::bot())
    }
}

/// Marker for algebras whose `⊕` is *selective*: `a ⊕ b ∈ {a, b}`.
///
/// On `[0, 1]` the only selective t-conorm is `max`, so a `SelectiveOr`
/// algebra pairs its t-norm with `max` — and by monotonicity every t-norm
/// distributes over `max`, making each such algebra a commutative semiring
/// (the `*`-with-max family: fuzzy `(min, max)`, Viterbi `(·, max)`,
/// bounded-sum-with-max). Selectivity is what makes single-derivation
/// witnesses exact: a `max` always realizes one operand, so every answer
/// degree is carried by one derivation ([`crate::provenance`]'s bound).
/// Implemented by [`Godel`] and [`Viterbi`].
pub trait SelectiveOr: Truth {}

impl SelectiveOr for Godel {}
impl SelectiveOr for Viterbi {}

/// Marker for algebras whose `⊗` and `⊕` are both idempotent.
///
/// By the classical t-norm result (Klement, Mesiar & Pap, *Triangular
/// Norms*), the only idempotent t-norm is `min` and the only idempotent
/// t-conorm is `max` — so an `Idempotent` algebra IS `(min, max)`, i.e. the
/// fuzzy semiring of Green et al. (see the module docs), where every answer
/// degree is a single bottleneck derivation. Purely descriptive today: no
/// API bounds on it ([`crate::provenance`] is bounded by the weaker
/// [`SelectiveOr`], which Viterbi also satisfies). Implemented by [`Godel`]
/// alone — by the cited uniqueness result nothing else lawfully can be.
pub trait Idempotent: SelectiveOr {}

impl Idempotent for Godel {}

/// Gödel algebra: the minimum t-norm. The canonical **Heyting algebra** on
/// `[0, 1]`; its negation is the crisp intuitionistic pseudo-complement.
///
/// Best for intersections — `min` is the largest t-norm, so it preserves degree
/// through AND.
#[derive(Debug, Clone, Copy, Default)]
pub struct Godel;

impl Truth for Godel {
    #[inline]
    fn and(a: f32, b: f32) -> f32 {
        a.min(b)
    }
    #[inline]
    fn or(a: f32, b: f32) -> f32 {
        a.max(b)
    }
    #[inline]
    fn residuum(a: f32, b: f32) -> f32 {
        if a <= b {
            1.0
        } else {
            b
        }
    }
}

/// Product (Goguen) algebra: the product t-norm `a · b`, with the probabilistic
/// sum as its t-conorm. Negation is crisp.
///
/// Best for chains — multiplying degrees through hops propagates magnitude,
/// distinguishing a long confident path from a short weak one.
#[derive(Debug, Clone, Copy, Default)]
pub struct Product;

impl Truth for Product {
    #[inline]
    fn and(a: f32, b: f32) -> f32 {
        a * b
    }
    #[inline]
    fn or(a: f32, b: f32) -> f32 {
        a + b - a * b
    }
    #[inline]
    fn residuum(a: f32, b: f32) -> f32 {
        if a <= b {
            1.0
        } else {
            // a > b ≥ 0 implies a > 0, so the division is safe.
            b / a
        }
    }
}

/// Viterbi algebra: the product t-norm paired with `max` — the classical
/// Viterbi semiring `([0, 1], max, ·, 0, 1)` of best-derivation decoding.
///
/// Chains propagate magnitude like [`Product`], but disjunction picks the
/// strongest alternative instead of accumulating, which keeps the algebra a
/// semiring (provenance machinery applies) and makes it witnessable
/// ([`SelectiveOr`]): the answer degree is exactly the best derivation's
/// product. The residuum is the Goguen residuum (it depends only on `⊗`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Viterbi;

impl Truth for Viterbi {
    #[inline]
    fn and(a: f32, b: f32) -> f32 {
        a * b
    }
    #[inline]
    fn or(a: f32, b: f32) -> f32 {
        a.max(b)
    }
    #[inline]
    fn residuum(a: f32, b: f32) -> f32 {
        if a <= b {
            1.0
        } else {
            b / a
        }
    }
}

/// Łukasiewicz algebra: the bounded-sum t-norm `max(0, a + b − 1)`. The
/// involutive MV-algebra whose negation is `1 − a`.
///
/// Best for negation-bearing queries: `¬a = 1 − a` ranks "not X" smoothly,
/// matching BetaE / `tranz::query`. The genuine Heyting negation
/// ([`Godel`]/[`Product`]) is crisp and cannot rank.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lukasiewicz;

impl Truth for Lukasiewicz {
    #[inline]
    fn and(a: f32, b: f32) -> f32 {
        (a + b - 1.0).max(0.0)
    }
    #[inline]
    fn or(a: f32, b: f32) -> f32 {
        (a + b).min(1.0)
    }
    #[inline]
    fn residuum(a: f32, b: f32) -> f32 {
        (1.0 - a + b).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const EPS: f32 = 1e-5;

    /// A degree in `[0, 1]`, the domain of every truth algebra.
    fn degree() -> impl Strategy<Value = f32> {
        (0.0f32..=1.0).prop_map(|x| (x * 1000.0).round() / 1000.0)
    }

    // Each algebra is checked against the residuated-lattice laws. Generic over
    // the marker type so the same laws police all three.
    fn check_unit<T: Truth>(a: f32) {
        assert!((T::and(a, T::top()) - a).abs() < EPS, "unit ⊤ failed");
        assert!(T::and(a, T::bot()).abs() < EPS, "annihilator ⊥ failed");
    }

    fn check_commutative<T: Truth>(a: f32, b: f32) {
        assert!((T::and(a, b) - T::and(b, a)).abs() < EPS);
        assert!((T::or(a, b) - T::or(b, a)).abs() < EPS);
    }

    fn check_associativity<T: Truth>(a: f32, b: f32, c: f32) {
        assert!(
            (T::and(T::and(a, b), c) - T::and(a, T::and(b, c))).abs() < EPS,
            "⊗ not associative"
        );
        assert!(
            (T::or(T::or(a, b), c) - T::or(a, T::or(b, c))).abs() < EPS,
            "⊕ not associative"
        );
    }

    fn check_or_identities<T: Truth>(a: f32) {
        // Dual to ⊗'s unit/annihilator: ⊕ has unit ⊥ and annihilator ⊤.
        assert!((T::or(a, T::bot()) - a).abs() < EPS, "or unit ⊥ failed");
        assert!(
            (T::or(a, T::top()) - T::top()).abs() < EPS,
            "or annihilator ⊤ failed"
        );
    }

    fn check_de_morgan<T: Truth>(a: f32, b: f32) {
        // ¬(a ⊗ b) = ¬a ⊕ ¬b. Holds for the involutive Łukasiewicz negation,
        // and — because [0, 1] under any of these t-norms is a chain — for the
        // crisp Gödel/Product pseudo-complement as well (min/product is 0 iff a
        // factor is 0, which is exactly when ¬a ⊕ ¬b saturates to ⊤).
        assert!(
            (T::neg(T::and(a, b)) - T::or(T::neg(a), T::neg(b))).abs() < EPS,
            "De Morgan ¬(a ⊗ b) = ¬a ⊕ ¬b failed"
        );
    }

    fn check_residuation<T: Truth>(a: f32, b: f32, c: f32) {
        // Modus ponens: a ⊗ (a → b) ≤ b.
        assert!(
            T::and(a, T::residuum(a, b)) <= b + EPS,
            "modus ponens failed: {a} ⊗ ({a}→{b}) > {b}"
        );
        // Adjunction unit: c ≤ a → (a ⊗ c).
        assert!(
            c <= T::residuum(a, T::and(a, c)) + EPS,
            "adjunction unit failed for a={a}, c={c}"
        );
        // Reflexivity: a ≤ b ⟹ a → b = ⊤.
        if a <= b {
            assert!(
                (T::residuum(a, b) - T::top()).abs() < EPS,
                "residuum not ⊤ when {a} ≤ {b}"
            );
        }
    }

    proptest! {
        #[test]
        fn godel_is_a_residuated_lattice(a in degree(), b in degree(), c in degree()) {
            check_unit::<Godel>(a);
            check_commutative::<Godel>(a, b);
            check_associativity::<Godel>(a, b, c);
            check_or_identities::<Godel>(a);
            check_de_morgan::<Godel>(a, b);
            check_residuation::<Godel>(a, b, c);
        }

        #[test]
        fn product_is_a_residuated_lattice(a in degree(), b in degree(), c in degree()) {
            check_unit::<Product>(a);
            check_commutative::<Product>(a, b);
            check_associativity::<Product>(a, b, c);
            check_or_identities::<Product>(a);
            check_de_morgan::<Product>(a, b);
            check_residuation::<Product>(a, b, c);
        }

        #[test]
        fn lukasiewicz_is_a_residuated_lattice(a in degree(), b in degree(), c in degree()) {
            check_unit::<Lukasiewicz>(a);
            check_commutative::<Lukasiewicz>(a, b);
            check_associativity::<Lukasiewicz>(a, b, c);
            check_or_identities::<Lukasiewicz>(a);
            check_de_morgan::<Lukasiewicz>(a, b);
            check_residuation::<Lukasiewicz>(a, b, c);
        }

        #[test]
        fn viterbi_is_a_residuated_lattice(a in degree(), b in degree(), c in degree()) {
            check_unit::<Viterbi>(a);
            check_commutative::<Viterbi>(a, b);
            check_associativity::<Viterbi>(a, b, c);
            check_or_identities::<Viterbi>(a);
            check_de_morgan::<Viterbi>(a, b);
            check_residuation::<Viterbi>(a, b, c);
        }

        #[test]
        fn selective_or_algebras_distribute(a in degree(), b in degree(), c in degree()) {
            // The semiring law the SelectiveOr marker promises: ⊗ over ⊕.
            prop_assert!((Godel::and(a, Godel::or(b, c))
                - Godel::or(Godel::and(a, b), Godel::and(a, c))).abs() < EPS);
            prop_assert!((Viterbi::and(a, Viterbi::or(b, c))
                - Viterbi::or(Viterbi::and(a, b), Viterbi::and(a, c))).abs() < EPS);
        }

        #[test]
        fn lukasiewicz_negation_is_involutive(a in degree()) {
            // ¬¬a = a, and ¬a = 1 − a.
            prop_assert!((Lukasiewicz::neg(a) - (1.0 - a)).abs() < EPS);
            prop_assert!((Lukasiewicz::neg(Lukasiewicz::neg(a)) - a).abs() < EPS);
        }

        #[test]
        fn godel_negation_is_the_crisp_pseudo_complement(a in degree()) {
            // ¬a = 1 if a = 0 else 0 — the intuitionistic pseudo-complement.
            let expected = if a <= EPS { 1.0 } else { 0.0 };
            prop_assert!((Godel::neg(a) - expected).abs() < EPS);
        }
    }
}
