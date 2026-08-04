//! ragvault-engine: the embedded storage + retrieval engine.
//!
//! Combines the document model, WAL durability, snapshot persistence and the
//! vector/lexical indexes into a single-writer, multi-reader vault. This is
//! what the Python bindings expose.

pub mod engine;
pub mod segment;
pub mod snapshot;
pub mod wal;

pub use engine::{EngineConfig, SearchHit, SearchRequest, VaultEngine};
/// Re-exported so the Python layer shares the *same* tokenizer the BM25
/// index uses. Two implementations of "what is a word" would drift, and a
/// drift between how the index tokenizes and how dedup tokenizes is
/// invisible until it misbehaves.
pub use ragvault_retrieval::tokenize;
