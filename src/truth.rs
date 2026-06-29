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
            check_residuation::<Godel>(a, b, c);
        }

        #[test]
        fn product_is_a_residuated_lattice(a in degree(), b in degree(), c in degree()) {
            check_unit::<Product>(a);
            check_commutative::<Product>(a, b);
            check_residuation::<Product>(a, b, c);
        }

        #[test]
        fn lukasiewicz_is_a_residuated_lattice(a in degree(), b in degree(), c in degree()) {
            check_unit::<Lukasiewicz>(a);
            check_commutative::<Lukasiewicz>(a, b);
            check_residuation::<Lukasiewicz>(a, b, c);
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
