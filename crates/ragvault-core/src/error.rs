use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for all RagVault crates.
///
/// Every variant carries enough context for the Python layer to build an
/// actionable exception message (operation, expected vs. received, hint).
#[derive(Debug, Error)]
pub enum Error {
    #[error("io error during {op}: {source}")]
    Io {
        op: String,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid input for {field}: expected {expected}, got {got}")]
    InvalidInput {
        field: String,
        expected: String,
        got: String,
    },

    #[error("dimension mismatch: index has dimension {expected}, vector has {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("document not found: {0}")]
    DocumentNotFound(String),

    #[error("chunk not found: {0}")]
    ChunkNotFound(String),

    #[error("corrupt data in {path}: {detail}")]
    Corrupt { path: String, detail: String },

    #[error("incompatible format version {found} (supported: {supported}) in {path}")]
    IncompatibleFormat {
        path: String,
        found: u32,
        supported: u32,
    },

    #[error("vault is locked by another writer: {path} (owner: {owner})")]
    Locked { path: String, owner: String },

    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn io(op: impl Into<String>, source: std::io::Error) -> Self {
        Error::Io {
            op: op.into(),
            source,
        }
    }

    pub fn invalid(
        field: impl Into<String>,
        expected: impl Into<String>,
        got: impl Into<String>,
    ) -> Self {
        Error::InvalidInput {
            field: field.into(),
            expected: expected.into(),
            got: got.into(),
        }
    }

    pub fn corrupt(path: impl Into<String>, detail: impl Into<String>) -> Self {
        Error::Corrupt {
            path: path.into(),
            detail: detail.into(),
        }
    }
}
