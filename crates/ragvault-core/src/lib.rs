//! ragvault-core: shared types, errors, the RAG-native document model and the
//! metadata filter DSL. This crate has no I/O, no Python, no CUDA and no
//! network dependencies — every other crate builds on top of it.

pub mod error;
pub mod filter;
pub mod types;

pub use error::{Error, Result};
pub use filter::Filter;
pub use types::{now_millis, Chunk, Document, DocumentVersion, Metric, Source, SparseVector};
