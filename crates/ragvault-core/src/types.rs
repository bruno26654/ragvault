use serde::{Deserialize, Serialize};

/// Milliseconds since the Unix epoch.
pub type Timestamp = u64;

/// A logical origin for documents (a directory, a URL, an upload, ...).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    pub source_id: String,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub content_hash: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub created_at: Timestamp,
    #[serde(default)]
    pub updated_at: Timestamp,
}

/// A document. The engine keeps exactly one *current* version visible to
/// queries; replacement publishes a new version atomically.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub document_id: String,
    #[serde(default)]
    pub source_id: Option<String>,
    pub current_version: u64,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// An immutable snapshot of a document's content at a given version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentVersion {
    pub document_id: String,
    pub version: u64,
    pub content_hash: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub created_at: Timestamp,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// The retrieval unit. Chunks preserve provenance (document, version,
/// position, section, page) and neighborhood links for context expansion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    pub chunk_id: String,
    pub document_id: String,
    pub document_version: u64,
    pub chunk_index: u32,
    pub text: String,
    #[serde(default)]
    pub byte_start: Option<u64>,
    #[serde(default)]
    pub byte_end: Option<u64>,
    #[serde(default)]
    pub token_start: Option<u64>,
    #[serde(default)]
    pub token_end: Option<u64>,
    #[serde(default)]
    pub token_count: Option<u32>,
    #[serde(default)]
    pub page_number: Option<u32>,
    #[serde(default)]
    pub section_path: Vec<String>,
    #[serde(default)]
    pub previous_chunk_id: Option<String>,
    #[serde(default)]
    pub next_chunk_id: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A sparse vector (e.g. SPLADE / learned lexical representations).
/// Indices must be strictly increasing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn validate(&self) -> crate::Result<()> {
        if self.indices.len() != self.values.len() {
            return Err(crate::Error::invalid(
                "sparse_vector",
                "indices.len() == values.len()",
                format!("{} != {}", self.indices.len(), self.values.len()),
            ));
        }
        for w in self.indices.windows(2) {
            if w[0] >= w[1] {
                return Err(crate::Error::invalid(
                    "sparse_vector.indices",
                    "strictly increasing indices",
                    format!("{} >= {}", w[0], w[1]),
                ));
            }
        }
        Ok(())
    }
}

/// Distance/similarity metric for dense vector fields.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Cosine similarity (vectors are normalized at insert time; score is dot).
    #[default]
    Cosine,
    /// Raw inner product (higher is better).
    Dot,
    /// Euclidean distance (lower is better; scores are negated distances).
    L2,
}

pub fn now_millis() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_vector_validation() {
        let ok = SparseVector {
            indices: vec![1, 5, 9],
            values: vec![0.5, 1.0, 0.25],
        };
        assert!(ok.validate().is_ok());

        let bad_len = SparseVector {
            indices: vec![1, 2],
            values: vec![0.5],
        };
        assert!(bad_len.validate().is_err());

        let bad_order = SparseVector {
            indices: vec![5, 1],
            values: vec![0.5, 1.0],
        };
        assert!(bad_order.validate().is_err());
    }

    #[test]
    fn chunk_roundtrip() {
        let chunk = Chunk {
            chunk_id: "c1".into(),
            document_id: "d1".into(),
            document_version: 1,
            chunk_index: 0,
            text: "hello".into(),
            byte_start: Some(0),
            byte_end: Some(5),
            token_start: None,
            token_end: None,
            token_count: Some(1),
            page_number: None,
            section_path: vec!["Intro".into()],
            previous_chunk_id: None,
            next_chunk_id: Some("c2".into()),
            metadata: serde_json::json!({"lang": "en"}),
        };
        let s = serde_json::to_string(&chunk).unwrap();
        let back: Chunk = serde_json::from_str(&s).unwrap();
        assert_eq!(chunk, back);
    }
}
