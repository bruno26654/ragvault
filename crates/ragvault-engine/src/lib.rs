//! ragvault-engine: the embedded storage + retrieval engine.
//!
//! Combines the document model, WAL durability, snapshot persistence and the
//! vector/lexical indexes into a single-writer, multi-reader vault. This is
//! what the Python bindings expose.

pub mod engine;
pub mod snapshot;
pub mod wal;

pub use engine::{EngineConfig, SearchHit, SearchRequest, VaultEngine};
