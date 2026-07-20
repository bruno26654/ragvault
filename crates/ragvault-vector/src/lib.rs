//! ragvault-vector: CPU vector kernels and index backends (Flat, HNSW).
//!
//! Kernels are written as simple chunked scalar loops that LLVM
//! auto-vectorizes (SSE/AVX2 on x86-64, NEON on aarch64) while staying
//! portable — published wheels never require `target-cpu=native`.
//! Every optimized path has a scalar reference and differential tests.

pub mod arena;
pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod kernels;
pub mod sq8;
pub mod topk;

pub use arena::VectorArena;
pub use flat::FlatIndex;
pub use hnsw::{Hnsw, HnswConfig};
pub use ivf::{IvfConfig, IvfIndex};
pub use sq8::Sq8Arena;
pub use topk::TopK;
