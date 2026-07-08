//! Adapters from ecosystem models to [`AtomicScorer`](crate::AtomicScorer).
//!
//! Adapters that pull an ecosystem crate sit behind an optional feature so the
//! core stays dependency-free; the self-contained ones are always available.
//! They are the worked proof that the [`AtomicScorer`](crate::AtomicScorer) seam
//! carries real trained embeddings, not just the in-memory [`crate::FuzzyKg`].
//!
//! Layout: `point` wraps tranz scorers and `temporal_point` tranz temporal
//! scorers with timestamp-set hops (feature `tranz`); `box_model` is the
//! Query2Box-style atomic scorer over trained boxes and `box_dnf` its
//! geometric execution mode (feature `subsume`); `faithful_box` is the faithful
//! EL++ counterpart (entities are boxes, containment is subsumption) and needs
//! no ecosystem crate, so it is always available.

#[cfg(feature = "subsume")]
mod box_dnf;
#[cfg(feature = "subsume")]
mod box_model;
#[cfg(any(feature = "hyperball", feature = "precinct", feature = "vicinity"))]
mod candidates;
mod faithful_box;
#[cfg(feature = "tranz")]
mod point;
#[cfg(feature = "tranz")]
mod temporal_point;

#[cfg(feature = "subsume")]
pub use box_dnf::{BoxDnf, Explanation, MaterializeError, QueryBox};
#[cfg(feature = "subsume")]
pub use box_model::{BoxModel, BoxModelError};
#[cfg(feature = "hyperball")]
pub use candidates::HyperbolicKnnCandidates;
#[cfg(feature = "precinct")]
pub use candidates::PrecinctSubsumers;
#[cfg(feature = "vicinity")]
pub use candidates::VicinityCandidates;
pub use faithful_box::{FaithfulBoxError, FaithfulBoxModel};
#[cfg(feature = "tranz")]
pub use point::PointModel;
#[cfg(feature = "tranz")]
pub use temporal_point::TemporalPointModel;
