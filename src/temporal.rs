//! Temporal knowledge graphs: time-scoped hops for the query engine.
//!
//! Facts in a temporal KG carry a validity interval (`president_of` from
//! 1993 to 2001; a point event has `start == end`). The complex-query
//! literature scopes hops with temporal operators — before, after, between
//! (TFLEX; and interval-native embeddings like HGE use Allen-style interval
//! relations). This module makes those operators available to the existing
//! engine without touching it: a [`TimeWindow`] is registered against a base
//! relation on the [`TemporalKg`], which returns a **virtual relation id**;
//! a hop through that id scores only the facts whose validity interval
//! satisfies the window. Time-scoped queries are then ordinary [`Query`]
//! DAGs, so intersection planning, candidate pruning
//! ([`CandidateSource`]), conformal
//! calibration, and the easy/hard evaluation split all apply to temporal
//! queries with no new machinery.
//!
//! The classic query this enables — "who held office before τ AND after τ'"
//! (an entity with two non-adjacent terms) — is an ordinary
//! [`Query::intersection`] of two hops through differently-windowed virtual
//! relations; see `examples/temporal_query.rs`.
//!
//! Windows are crisp (a fact either satisfies the window or not; degrees
//! come from the fact's weight). Soft window boundaries are a scorer-side
//! refinement, same as soft literal ramps for [`Query::given`]. Windows can
//! also be anchored to another fact's validity interval — TFLEX's
//! event-relative before/after/during operators — via
//! [`TemporalKg::windowed_after_fact`] and siblings; the full TFLEX design
//! (fuzzy sets over a timestamp sort, jointly with the entity sort) remains
//! out of scope until a temporal scorer exists to demand it.
//!
//! [`Query`]: crate::Query
//! [`Query::given`]: crate::Query::given
//! [`Query::intersection`]: crate::Query::intersection

use std::collections::HashMap;

use crate::prune::CandidateSource;
use crate::query::AtomicScorer;

/// A set of discrete timestamp ids over a fixed axis `0..num_timestamps`.
///
/// The carrier for TFLEX-style timestamp-set logic over event KGs (facts
/// stamped with a day id, as in ICEWS): where [`TimeWindow`] is a predicate
/// over continuous validity intervals, `TimeSet` is an explicit set, closed
/// under union, intersection, and complement — which windows are not
/// (the complement of an interval is two rays, the union of two windows is
/// non-contiguous). The window vocabulary survives as constructors:
/// [`before`](Self::before), [`after`](Self::after),
/// [`between`](Self::between) build the contiguous special cases.
///
/// Backed by a bitset; ICEWS14's axis is 365 days, so a set is 6 words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSet {
    blocks: Vec<u64>,
    n: usize,
}

impl TimeSet {
    /// The empty set over an axis of `n` timestamps.
    pub fn empty(n: usize) -> Self {
        Self {
            blocks: vec![0; n.div_ceil(64)],
            n,
        }
    }

    /// The full axis `0..n`.
    pub fn all(n: usize) -> Self {
        let mut s = Self::empty(n);
        for t in 0..n {
            s.insert(t);
        }
        s
    }

    /// Timestamps strictly before `t` (clamped to the axis).
    pub fn before(t: usize, n: usize) -> Self {
        let mut s = Self::empty(n);
        for i in 0..t.min(n) {
            s.insert(i);
        }
        s
    }

    /// Timestamps strictly after `t`.
    pub fn after(t: usize, n: usize) -> Self {
        let mut s = Self::empty(n);
        for i in t.saturating_add(1)..n {
            s.insert(i);
        }
        s
    }

    /// Timestamps in `[a, b]` inclusive (clamped to the axis).
    pub fn between(a: usize, b: usize, n: usize) -> Self {
        let mut s = Self::empty(n);
        for i in a..=b.min(n.saturating_sub(1)) {
            if i < n {
                s.insert(i);
            }
        }
        s
    }

    /// The single timestamp `t` (empty if `t` is off-axis).
    pub fn singleton(t: usize, n: usize) -> Self {
        let mut s = Self::empty(n);
        s.insert(t);
        s
    }

    /// Axis size this set is defined over.
    pub fn num_timestamps(&self) -> usize {
        self.n
    }

    /// Add a timestamp (ignored if off-axis).
    pub fn insert(&mut self, t: usize) {
        if t < self.n {
            self.blocks[t / 64] |= 1 << (t % 64);
        }
    }

    /// Is `t` in the set?
    pub fn contains(&self, t: usize) -> bool {
        t < self.n && self.blocks[t / 64] & (1 << (t % 64)) != 0
    }

    /// Number of timestamps in the set.
    pub fn len(&self) -> usize {
        self.blocks.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Is the set empty?
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().all(|&b| b == 0)
    }

    /// Set union. Both sets must share the axis.
    ///
    /// # Panics
    /// Panics if the axes differ.
    pub fn union(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "TimeSet axes differ");
        Self {
            blocks: self
                .blocks
                .iter()
                .zip(&other.blocks)
                .map(|(a, b)| a | b)
                .collect(),
            n: self.n,
        }
    }

    /// Set intersection. Both sets must share the axis.
    ///
    /// # Panics
    /// Panics if the axes differ.
    pub fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "TimeSet axes differ");
        Self {
            blocks: self
                .blocks
                .iter()
                .zip(&other.blocks)
                .map(|(a, b)| a & b)
                .collect(),
            n: self.n,
        }
    }

    /// Complement within the axis (trailing off-axis bits stay clear).
    pub fn complement(&self) -> Self {
        let mut blocks: Vec<u64> = self.blocks.iter().map(|b| !b).collect();
        let tail = self.n % 64;
        if tail != 0 {
            if let Some(last) = blocks.last_mut() {
                *last &= (1u64 << tail) - 1;
            }
        }
        Self { blocks, n: self.n }
    }

    /// Iterate the member timestamps in increasing order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n).filter(move |&t| self.contains(t))
    }
}

/// A predicate over a fact's validity interval `[start, end]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeWindow {
    /// The fact ended strictly before `t`.
    Before(f64),
    /// The fact started strictly after `t`.
    After(f64),
    /// The fact's validity intersects `[a, b]` (inclusive).
    Between(f64, f64),
    /// No temporal constraint (the base relation's plain semantics).
    AnyTime,
}

/// One weighted fact tail with its validity interval.
#[derive(Debug, Clone, Copy)]
struct Fact {
    tail: usize,
    start: f64,
    end: f64,
    weight: f32,
}

impl TimeWindow {
    /// Does a fact valid over `[start, end]` satisfy this window?
    pub fn admits(&self, start: f64, end: f64) -> bool {
        match *self {
            TimeWindow::Before(t) => end < t,
            TimeWindow::After(t) => start > t,
            TimeWindow::Between(a, b) => start <= b && end >= a,
            TimeWindow::AnyTime => true,
        }
    }
}

/// A fuzzy temporal knowledge graph: weighted facts with validity intervals,
/// plus a registry of time-windowed virtual relations.
///
/// Base relations occupy ids `0..n_relations`; [`windowed`](Self::windowed)
/// registers `(base relation, window)` pairs at fresh ids beyond them. Both
/// kinds evaluate through the [`AtomicScorer`] impl (a base relation is
/// [`TimeWindow::AnyTime`]), and the [`CandidateSource`] impl proposes the
/// exact tails of each (possibly windowed) hop, so the pruned path is exact
/// on this graph.
#[derive(Debug, Clone, Default)]
pub struct TemporalKg {
    n_entities: usize,
    n_relations: usize,
    /// Facts keyed by `(head, base relation)`.
    facts: HashMap<(usize, usize), Vec<Fact>>,
    /// Virtual relation registry, indexed by `id - n_relations`.
    windows: Vec<(usize, TimeWindow)>,
}

impl TemporalKg {
    /// An empty graph over `n_entities` entities and `n_relations` base
    /// relations (ids `0..n_relations`).
    pub fn new(n_entities: usize, n_relations: usize) -> Self {
        Self {
            n_entities,
            n_relations,
            facts: HashMap::new(),
            windows: Vec::new(),
        }
    }

    /// Add a fact `(head, relation, tail)` valid over `[start, end]` with
    /// membership `weight` (clamped to `[0, 1]`). Out-of-range entity or
    /// relation ids are ignored; a reversed interval is normalized.
    pub fn add_fact(
        &mut self,
        head: usize,
        relation: usize,
        tail: usize,
        start: f64,
        end: f64,
        weight: f32,
    ) {
        if head >= self.n_entities || tail >= self.n_entities || relation >= self.n_relations {
            return;
        }
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        self.facts.entry((head, relation)).or_default().push(Fact {
            tail,
            start,
            end,
            weight: weight.clamp(0.0, 1.0),
        });
    }

    /// Register a time-scoped view of `relation` and return its virtual
    /// relation id, usable anywhere a relation id goes
    /// ([`Query::anchor`](crate::Query::anchor), `then`, candidates).
    /// Returns `None` if `relation` is not a base relation.
    pub fn windowed(&mut self, relation: usize, window: TimeWindow) -> Option<usize> {
        if relation >= self.n_relations {
            return None;
        }
        self.windows.push((relation, window));
        Some(self.n_relations + self.windows.len() - 1)
    }

    /// Resolve a (possibly virtual) relation id to `(base, window)`.
    fn resolve(&self, relation: usize) -> Option<(usize, TimeWindow)> {
        if relation < self.n_relations {
            Some((relation, TimeWindow::AnyTime))
        } else {
            self.windows.get(relation - self.n_relations).copied()
        }
    }

    /// Facts for `(anchor, relation)` admitted by the id's window.
    fn admitted(&self, anchor: usize, relation: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.resolve(relation)
            .into_iter()
            .flat_map(move |(base, window)| {
                self.facts
                    .get(&(anchor, base))
                    .into_iter()
                    .flatten()
                    .filter(move |f| window.admits(f.start, f.end))
                    .map(|f| (f.tail, f.weight))
            })
    }
}

impl TemporalKg {
    /// The validity hull of the facts `(head, relation, tail)`: the earliest
    /// start and latest end over matching facts, or `None` when no such fact
    /// exists. The anchor for event-relative windows.
    pub fn fact_interval(&self, head: usize, relation: usize, tail: usize) -> Option<(f64, f64)> {
        let facts = self.facts.get(&(head, relation))?;
        let mut hull: Option<(f64, f64)> = None;
        for f in facts.iter().filter(|f| f.tail == tail) {
            hull = Some(match hull {
                None => (f.start, f.end),
                Some((s, e)) => (s.min(f.start), e.max(f.end)),
            });
        }
        hull
    }

    /// Register `relation` scoped to strictly after the referenced fact's
    /// validity (TFLEX's after-event operator). `None` if the relation or
    /// the fact is unknown.
    pub fn windowed_after_fact(
        &mut self,
        relation: usize,
        event: (usize, usize, usize),
    ) -> Option<usize> {
        let (_, end) = self.fact_interval(event.0, event.1, event.2)?;
        self.windowed(relation, TimeWindow::After(end))
    }

    /// Register `relation` scoped to strictly before the referenced fact's
    /// validity (the before-event operator).
    pub fn windowed_before_fact(
        &mut self,
        relation: usize,
        event: (usize, usize, usize),
    ) -> Option<usize> {
        let (start, _) = self.fact_interval(event.0, event.1, event.2)?;
        self.windowed(relation, TimeWindow::Before(start))
    }

    /// Register `relation` scoped to overlap the referenced fact's validity
    /// (the during-event operator).
    pub fn windowed_during_fact(
        &mut self,
        relation: usize,
        event: (usize, usize, usize),
    ) -> Option<usize> {
        let (start, end) = self.fact_interval(event.0, event.1, event.2)?;
        self.windowed(relation, TimeWindow::Between(start, end))
    }
}

impl AtomicScorer for TemporalKg {
    fn num_entities(&self) -> usize {
        self.n_entities
    }

    fn project(&self, anchor: usize, relation: usize) -> Vec<f32> {
        let mut scores = vec![0.0_f32; self.n_entities];
        for (t, w) in self.admitted(anchor, relation) {
            if t < self.n_entities && w > scores[t] {
                scores[t] = w; // parallel facts take the max, as in FuzzyKg.
            }
        }
        scores
    }
}

impl CandidateSource for TemporalKg {
    fn candidates(&self, anchor: usize, relation: usize) -> Option<Vec<usize>> {
        Some(self.admitted(anchor, relation).map(|(t, _)| t).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{answer_query, answer_query_topk};
    use crate::{answer_query_topk_pruned, Godel, Query, QueryConfig};

    /// Entities: 0=alice 1=bob 2=carol 3=office. Relation 0 = holds(office).
    /// alice: one term [1993, 2001]. bob: two terms [1985, 1989] and
    /// [2005, 2009]. carol: one term [2017, 2021].
    fn kg() -> TemporalKg {
        let mut kg = TemporalKg::new(4, 1);
        kg.add_fact(3, 0, 0, 1993.0, 2001.0, 1.0);
        kg.add_fact(3, 0, 1, 1985.0, 1989.0, 1.0);
        kg.add_fact(3, 0, 1, 2005.0, 2009.0, 1.0);
        kg.add_fact(3, 0, 2, 2017.0, 2021.0, 1.0);
        kg
    }

    /// TimeSet algebra: double complement is identity, De Morgan holds, and
    /// before/singleton/after partition the axis. Checked on an axis that
    /// crosses a word boundary (n = 70) so tail masking is exercised.
    #[test]
    fn timeset_algebra_laws() {
        let n = 70;
        let a = TimeSet::between(3, 40, n);
        let b = TimeSet::after(25, n);

        assert_eq!(a.complement().complement(), a);
        assert_eq!(
            a.union(&b).complement(),
            a.complement().intersect(&b.complement()),
            "De Morgan"
        );

        let t = 33;
        let partition = TimeSet::before(t, n)
            .union(&TimeSet::singleton(t, n))
            .union(&TimeSet::after(t, n));
        assert_eq!(partition, TimeSet::all(n));
        assert!(TimeSet::before(t, n)
            .intersect(&TimeSet::after(t, n))
            .is_empty());

        // Complement never leaks off-axis bits.
        assert_eq!(TimeSet::empty(n).complement(), TimeSet::all(n));
        assert_eq!(TimeSet::all(n).complement().len(), 0);
    }

    /// The TFLEX non-contiguous cases intervals cannot carry: a complement
    /// (two rays) and a union of two windows.
    #[test]
    fn timeset_represents_non_contiguous_sets() {
        let n = 100;
        let mid = TimeSet::between(40, 60, n);
        let rays = mid.complement();
        assert!(rays.contains(0) && rays.contains(99));
        assert!(!rays.contains(50));
        assert_eq!(rays.len(), 100 - 21);

        let two = TimeSet::between(0, 5, n).union(&TimeSet::between(90, 95, n));
        assert_eq!(two.len(), 12);
        assert!(!two.contains(50));
        let members: Vec<usize> = two.iter().collect();
        assert_eq!(members[0], 0);
        assert_eq!(*members.last().unwrap(), 95);
    }

    #[test]
    fn windows_admit_by_interval() {
        assert!(TimeWindow::Before(1990.0).admits(1985.0, 1989.0));
        assert!(!TimeWindow::Before(1989.0).admits(1985.0, 1989.0)); // strict
        assert!(TimeWindow::After(2004.0).admits(2005.0, 2009.0));
        assert!(!TimeWindow::After(2005.0).admits(2005.0, 2009.0)); // strict
        assert!(TimeWindow::Between(2000.0, 2006.0).admits(2005.0, 2009.0));
        assert!(TimeWindow::Between(2000.0, 2006.0).admits(1993.0, 2001.0));
        assert!(!TimeWindow::Between(2010.0, 2012.0).admits(2005.0, 2009.0));
    }

    /// Hand-oracle: "held the office before 1990" admits only bob's first
    /// term; "after 2010" only carol's.
    #[test]
    fn windowed_hops_scope_answers() {
        let mut kg = kg();
        let before_1990 = kg.windowed(0, TimeWindow::Before(1990.0)).unwrap();
        let after_2010 = kg.windowed(0, TimeWindow::After(2010.0)).unwrap();
        let cfg = QueryConfig::default();

        let s = answer_query::<Godel>(&kg, &Query::anchor(3, before_1990), &cfg);
        assert_eq!(s, vec![0.0, 1.0, 0.0, 0.0]);

        let s = answer_query::<Godel>(&kg, &Query::anchor(3, after_2010), &cfg);
        assert_eq!(s, vec![0.0, 0.0, 1.0, 0.0]);

        // The base relation is unconstrained: everyone who ever held it.
        let s = answer_query::<Godel>(&kg, &Query::anchor(3, 0), &cfg);
        assert_eq!(s, vec![1.0, 1.0, 1.0, 0.0]);
    }

    /// The TFLEX-motivating query: held office before 1990 AND after 2000 —
    /// two non-adjacent terms. Only bob qualifies, via an ordinary
    /// intersection of two windowed hops.
    #[test]
    fn two_terms_query_is_an_ordinary_intersection() {
        let mut kg = kg();
        let before_1990 = kg.windowed(0, TimeWindow::Before(1990.0)).unwrap();
        let after_2000 = kg.windowed(0, TimeWindow::After(2000.0)).unwrap();
        let cfg = QueryConfig::default();

        let q = Query::intersection(vec![
            Query::anchor(3, before_1990),
            Query::anchor(3, after_2000),
        ]);
        let top = answer_query_topk::<Godel>(&kg, &q, &cfg, 4);
        assert_eq!(top.first(), Some(&(1, 1.0)));
        assert!(top.iter().skip(1).all(|(_, d)| *d == 0.0));

        // And the pruned path agrees: TemporalKg is its own exact candidate
        // source, windows included.
        let pruned = answer_query_topk_pruned::<Godel>(&kg, &kg, &q, &cfg, 4);
        assert_eq!(pruned, vec![(1, 1.0)]);
    }

    /// Event-relative windows: "held the office after bob's FIRST term"
    /// admits alice (1993-2001) and carol but the window anchored to bob's
    /// hull (1985..2009) admits only carol.
    #[test]
    fn event_relative_windows_resolve_fact_hulls() {
        let mut kg = kg();
        // bob's hull spans both terms: [1985, 2009].
        assert_eq!(kg.fact_interval(3, 0, 1), Some((1985.0, 2009.0)));
        let after_bob = kg.windowed_after_fact(0, (3, 0, 1)).unwrap();
        let s = answer_query::<Godel>(&kg, &Query::anchor(3, after_bob), &cfg_default());
        assert_eq!(s, vec![0.0, 0.0, 1.0, 0.0], "only carol is after 2009");
        assert_eq!(kg.clone().windowed_after_fact(0, (3, 0, 9)), None);
    }

    fn cfg_default() -> QueryConfig {
        QueryConfig::default()
    }

    /// Unknown virtual ids and out-of-range base ids score nothing.
    #[test]
    fn unresolved_relations_score_zero() {
        let kg = kg();
        let cfg = QueryConfig::default();
        let s = answer_query::<Godel>(&kg, &Query::anchor(3, 99), &cfg);
        assert!(s.iter().all(|&d| d == 0.0));
        let mut kg2 = kg.clone();
        assert_eq!(kg2.windowed(7, TimeWindow::AnyTime), None);
    }
}
