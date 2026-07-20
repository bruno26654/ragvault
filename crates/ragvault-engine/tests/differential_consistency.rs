//! Differential consistency suite: for every backend configuration the
//! observable state must be identical
//!
//! ```text
//! before compact == after compact == after compact + reopen
//! ```
//!
//! across documents, versions, chunks, dense (flat/hnsw/sq8/ivf), BM25,
//! sparse, hybrid fusion, filters and tenants, after a workload of inserts,
//! replacements and deletes.
//!
//! Tie-break rule (documented): equal scores order by ascending internal id;
//! after compaction internal ids are re-assigned in ascending old-row order,
//! so relative order among ties is preserved. Score equality uses exact f32
//! comparison — compaction copies vectors byte-for-byte, so recomputed
//! scores must be bit-identical.

use ragvault_core::{Chunk, Document, SparseVector};
use ragvault_engine::{EngineConfig, SearchRequest, VaultEngine};
use serde_json::json;

fn doc(id: &str, tenant: &str) -> Document {
    Document {
        document_id: id.into(),
        source_id: None,
        current_version: 1,
        title: Some(format!("Title {id}")),
        metadata: json!({"tenant_id": tenant, "topic": format!("t{}", id.len() % 3)}),
    }
}

fn chunk(doc_id: &str, idx: u32, text: &str) -> Chunk {
    Chunk {
        chunk_id: format!("{doc_id}#{idx}"),
        document_id: doc_id.into(),
        document_version: 1,
        chunk_index: idx,
        text: text.into(),
        byte_start: None,
        byte_end: None,
        token_start: None,
        token_end: None,
        token_count: None,
        page_number: None,
        section_path: vec![],
        previous_chunk_id: None,
        next_chunk_id: None,
        metadata: json!({}),
    }
}

fn vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| (((seed * 31 + d * 7) % 97) as f32 / 97.0) - 0.5)
        .collect()
}

/// Apply a mixed workload: inserts, multi-chunk docs, sparse vectors,
/// replacements and deletes across two tenants.
fn workload(engine: &VaultEngine, dim: usize) {
    for i in 0..60 {
        let id = format!("doc{i}");
        let tenant = if i % 2 == 0 { "acme" } else { "globex" };
        let n_chunks = 1 + (i % 3);
        let chunks: Vec<Chunk> = (0..n_chunks)
            .map(|c| {
                chunk(
                    &id,
                    c as u32,
                    &format!("document {i} part {c} about subject{}", i % 7),
                )
            })
            .collect();
        let mut vectors = Vec::new();
        for c in 0..n_chunks {
            vectors.extend(vector(dim, i * 10 + c));
        }
        let sparse = if i % 4 == 0 {
            Some(
                (0..n_chunks)
                    .map(|c| {
                        Some(SparseVector {
                            indices: vec![(i % 11) as u32, 20 + c as u32],
                            values: vec![1.0 + i as f32 / 10.0, 0.5],
                        })
                    })
                    .collect(),
            )
        } else {
            None
        };
        engine
            .upsert_document(doc(&id, tenant), chunks, &vectors, sparse)
            .unwrap();
    }
    // Replace a third of the docs with new content (new versions).
    for i in (0..60).step_by(3) {
        let id = format!("doc{i}");
        let tenant = if i % 2 == 0 { "acme" } else { "globex" };
        engine
            .upsert_document(
                doc(&id, tenant),
                vec![chunk(&id, 0, &format!("replaced content {i} fresh"))],
                &vector(dim, 9000 + i),
                None,
            )
            .unwrap();
    }
    // Delete a fifth.
    for i in (0..60).step_by(5) {
        engine.delete_document(&format!("doc{i}")).unwrap();
    }
}

/// Snapshot of everything observable through the public API.
fn observe(engine: &VaultEngine, dim: usize) -> Vec<String> {
    let mut out = Vec::new();
    // Documents + versions + chunks.
    for d in engine.list_documents() {
        out.push(format!("doc {} v{}", d.document_id, d.current_version));
        for c in engine.get_document_chunks(&d.document_id) {
            out.push(format!(
                "chunk {} v{} idx{} text={}",
                c.chunk_id, c.document_version, c.chunk_index, c.text
            ));
        }
    }
    // Searches across signals, with and without filters/tenants.
    let queries: Vec<SearchRequest> = vec![
        SearchRequest {
            vector: Some(vector(dim, 123)),
            text: None,
            sparse: None,
            k: 10,
            mode: "dense".into(),
            candidates: None,
            filter: None,
            ef_search: Some(256),
            nprobe: Some(1024),
            weights: None,
        },
        SearchRequest {
            vector: None,
            text: Some("subject3 document part".into()),
            sparse: None,
            k: 10,
            mode: "keyword".into(),
            candidates: None,
            filter: None,
            ef_search: None,
            nprobe: None,
            weights: None,
        },
        SearchRequest {
            vector: None,
            text: None,
            sparse: Some(SparseVector {
                indices: vec![4, 21],
                values: vec![1.0, 1.0],
            }),
            k: 10,
            mode: "sparse".into(),
            candidates: None,
            filter: None,
            ef_search: None,
            nprobe: None,
            weights: None,
        },
        SearchRequest {
            vector: Some(vector(dim, 456)),
            text: Some("replaced content fresh".into()),
            sparse: None,
            k: 10,
            mode: "hybrid".into(),
            candidates: None,
            filter: Some(json!({"tenant_id": "acme"})),
            ef_search: Some(256),
            nprobe: Some(1024),
            weights: None,
        },
        SearchRequest {
            vector: Some(vector(dim, 789)),
            text: None,
            sparse: None,
            k: 10,
            mode: "dense".into(),
            candidates: None,
            filter: Some(json!({"topic": "t4"})),
            ef_search: Some(256),
            nprobe: Some(1024),
            weights: None,
        },
    ];
    for (qi, request) in queries.iter().enumerate() {
        let response = engine.search(request).unwrap();
        for hit in response.hits {
            out.push(format!(
                "q{qi} {} score={:.6} dense={:?} bm25={:?} sparse={:?}",
                hit.chunk_id,
                hit.score,
                hit.dense_score.map(|s| format!("{s:.6}")),
                hit.bm25_score.map(|s| format!("{s:.6}")),
                hit.sparse_score.map(|s| format!("{s:.6}")),
            ));
        }
    }
    out
}

fn run_differential(config: EngineConfig, label: &str) {
    let dim = config.dim;
    let dir = tempfile::tempdir().unwrap();
    let engine = VaultEngine::open(dir.path(), config.clone()).unwrap();
    workload(&engine, dim);
    // Flush so IVF (when configured) trains before the baseline observation.
    engine.flush().unwrap();
    let before = observe(&engine, dim);

    engine.compact().unwrap();
    let after_compact = observe(&engine, dim);
    assert_eq!(
        before, after_compact,
        "[{label}] compact changed observable state"
    );
    drop(engine);

    let engine = VaultEngine::open(dir.path(), config).unwrap();
    let after_reopen = observe(&engine, dim);
    assert_eq!(
        before, after_reopen,
        "[{label}] compact + reopen changed observable state"
    );
}

fn base_config(dim: usize) -> EngineConfig {
    let mut c = EngineConfig::new(dim);
    c.wal_sync = ragvault_engine::wal::SyncPolicy::Sync;
    c
}

#[test]
fn differential_flat() {
    let mut c = base_config(16);
    c.flat_threshold = 1_000_000;
    run_differential(c, "flat");
}

#[test]
fn differential_hnsw() {
    let mut c = base_config(16);
    c.flat_threshold = 10;
    run_differential(c, "hnsw");
}

#[test]
fn differential_sq8() {
    let mut c = base_config(16);
    c.quantization = "sq8".to_string();
    run_differential(c, "sq8");
}

#[test]
fn differential_ivf_flat() {
    let mut c = base_config(16);
    c.index = "ivf_flat".to_string();
    run_differential(c, "ivf_flat");
}

#[test]
fn differential_ivf_pq() {
    let mut c = base_config(16);
    c.index = "ivf_pq".to_string();
    run_differential(c, "ivf_pq");
}

#[test]
fn differential_mmap_reopen() {
    // memory workload; reopen under mmap must observe identical state.
    let dim = 16;
    let mut config = base_config(dim);
    config.flat_threshold = 1_000_000;
    let dir = tempfile::tempdir().unwrap();
    let engine = VaultEngine::open(dir.path(), config.clone()).unwrap();
    workload(&engine, dim);
    engine.flush().unwrap();
    let before = observe(&engine, dim);
    drop(engine);

    config.storage = "mmap".to_string();
    let engine = VaultEngine::open(dir.path(), config.clone()).unwrap();
    assert_eq!(before, observe(&engine, dim), "mmap reopen differs");
    engine.compact().unwrap();
    assert_eq!(before, observe(&engine, dim), "mmap compact differs");
}
