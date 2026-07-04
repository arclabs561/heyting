//! Adapters from ecosystem models to [`AtomicScorer`](crate::AtomicScorer).
//!
//! Each adapter is behind an optional feature so the core stays dependency-free.
//! They are the worked proof that the [`AtomicScorer`](crate::AtomicScorer) seam
//! carries real trained embeddings, not just the in-memory [`crate::FuzzyKg`].
//!
//! Layout: `point` wraps tranz scorers and `temporal_point` tranz temporal
//! scorers with timestamp-set hops (feature `tranz`); `box_model` is the
//! Query2Box-style atomic scorer over trained boxes and `box_dnf` its
//! geometric execution mode (feature `subsume`).

#[cfg(feature = "subsume")]
mod box_dnf;
#[cfg(feature = "subsume")]
mod box_model;
#[cfg(feature = "tranz")]
mod point;
#[cfg(feature = "tranz")]
mod temporal_point;

#[cfg(feature = "subsume")]
pub use box_dnf::{BoxDnf, Explanation, MaterializeError, QueryBox};
#[cfg(feature = "subsume")]
pub use box_model::{BoxModel, BoxModelError};
#[cfg(feature = "tranz")]
pub use point::PointModel;
#[cfg(feature = "tranz")]
pub use temporal_point::TemporalPointModel;
