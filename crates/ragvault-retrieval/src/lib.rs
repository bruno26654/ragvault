//! ragvault-retrieval: lexical retrieval (BM25), sparse retrieval and
//! candidate fusion.

pub mod bm25;
pub mod fusion;
pub mod sparse;

pub use bm25::{tokenize, Bm25Index, Bm25Params};
pub use fusion::{rrf_fuse, FusionInput};
pub use sparse::SparseIndex;
