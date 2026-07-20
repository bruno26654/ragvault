//! The vault engine: single-writer, multi-reader embedded store combining
//! the document model, WAL, snapshots and the dense/lexical/sparse indexes.
//!
//! Guarantees (v0.1, documented in docs/adr/0006):
//! - Atomicity per document operation: `upsert_document` / `delete_document`
//!   are WAL-logged before being applied; queries never observe a partially
//!   applied operation (the state lock is held for the whole apply).
//! - Snapshot-consistent queries: a search holds the read lock, so it sees
//!   one coherent state — no dirty reads, no mixed document versions.
//! - Recovery: reopen loads the last published snapshot generation and
//!   replays WAL records with `seq > manifest.seq`; replay is idempotent.
//! - Single writer per directory enforced with an exclusive file lock;
//!   multiple readers are supported within the process. Cross-process
//!   readers of a live vault are not supported in v0.1 (documented).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;

use ragvault_core::{now_millis, Chunk, Document, Error, Filter, Metric, Result, SparseVector};
use ragvault_retrieval::{rrf_fuse, Bm25Index, Bm25Params, FusionInput, SparseIndex};
use ragvault_vector::{
    ivf::MIN_TRAIN_ROWS, FlatIndex, Hnsw, HnswConfig, IvfConfig, IvfIndex, Sq8Arena, TopK,
    VectorArena,
};

use crate::snapshot::{self, PersistedStateV1, PersistedVersionV1};
use crate::wal::{SyncPolicy, Wal, WalOp};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineConfig {
    pub dim: usize,
    #[serde(default)]
    pub metric: Metric,
    #[serde(default)]
    pub hnsw: HnswConfig,
    #[serde(default)]
    pub bm25: Bm25Params,
    #[serde(default)]
    pub wal_sync: SyncPolicy,
    /// Below this many live vectors the planner always uses Flat.
    #[serde(default = "default_flat_threshold")]
    pub flat_threshold: usize,
    /// "none" (default) or "sq8". With sq8, dense search runs an int8
    /// quantized scan with f32 rescoring instead of building an HNSW graph
    /// (4x smaller scan, near-exact recall; see docs/PERFORMANCE.md).
    #[serde(default = "default_quantization")]
    pub quantization: String,
    /// Dense index selection: "auto" (default; hnsw, or sq8_flat when
    /// quantization="sq8"), "ivf_flat" or "ivf_pq". IVF indexes are
    /// rebuildable acceleration structures: trained on open/flush/compact,
    /// with fresh rows delta-scanned until the next rebuild.
    #[serde(default = "default_index")]
    pub index: String,
    #[serde(default)]
    pub ivf: IvfConfig,
    /// "memory" (default) or "mmap": serve snapshotted vectors from a
    /// read-only mmap of vectors.bin (page-cache resident); rows written
    /// after open live in RAM until the next flush+reopen. Runtime knob —
    /// it may differ between opens of the same vault.
    #[serde(default = "default_storage")]
    pub storage: String,
}

fn default_storage() -> String {
    "memory".to_string()
}

fn default_index() -> String {
    "auto".to_string()
}

fn default_quantization() -> String {
    "none".to_string()
}

fn default_flat_threshold() -> usize {
    1000
}

impl EngineConfig {
    pub fn new(dim: usize) -> Self {
        EngineConfig {
            dim,
            metric: Metric::Cosine,
            hnsw: HnswConfig::default(),
            bm25: Bm25Params::default(),
            wal_sync: SyncPolicy::default(),
            flat_threshold: default_flat_threshold(),
            quantization: default_quantization(),
            index: default_index(),
            ivf: IvfConfig::default(),
            storage: default_storage(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Dense query vector (required for dense/hybrid modes).
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    /// Query text (required for keyword/hybrid modes).
    #[serde(default)]
    pub text: Option<String>,
    /// Sparse query vector (optional signal).
    #[serde(default)]
    pub sparse: Option<SparseVector>,
    pub k: usize,
    /// "dense" | "keyword" | "sparse" | "hybrid" | "auto"
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Candidate pool per signal before fusion (defaults to 4*k, min 50).
    #[serde(default)]
    pub candidates: Option<usize>,
    #[serde(default)]
    pub filter: Option<serde_json::Value>,
    #[serde(default)]
    pub ef_search: Option<usize>,
    /// IVF lists to probe for this query (defaults to the config value).
    #[serde(default)]
    pub nprobe: Option<usize>,
    /// Fusion weights per signal.
    #[serde(default)]
    pub weights: Option<HashMap<String, f32>>,
}

fn default_mode() -> String {
    "auto".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub document_id: String,
    pub score: f32,
    pub dense_score: Option<f32>,
    pub bm25_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub internal_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    /// Explainable plan: backend chosen, reasons, parameters.
    pub plan: serde_json::Value,
}

struct StoredChunk {
    chunk: Chunk,
    /// Effective metadata for filtering: document metadata overlaid with
    /// chunk metadata plus injected provenance fields.
    eff_metadata: serde_json::Value,
}

struct State {
    config: EngineConfig,
    documents: HashMap<String, Document>,
    versions: HashMap<String, Vec<PersistedVersionV1>>,
    /// Row-aligned with the arena. `None` after compaction dropped a row.
    chunks: Vec<Option<StoredChunk>>,
    chunk_ids: HashMap<String, u32>,
    doc_rows: HashMap<String, Vec<u32>>,
    arena: VectorArena,
    sq8: Option<Sq8Arena>,
    ivf: Option<IvfIndex>,
    hnsw: Hnsw,
    bm25: Bm25Index,
    sparse: SparseIndex,
    seq: u64,
    wal: Wal,
    dirty: bool,
    generation: u64,
}

pub struct VaultEngine {
    path: PathBuf,
    state: RwLock<State>,
    _lock_file: fs::File,
}

fn effective_metadata(document: &Document, chunk: &Chunk) -> serde_json::Value {
    let mut merged = match &document.metadata {
        serde_json::Value::Object(m) => m.clone(),
        _ => serde_json::Map::new(),
    };
    if let serde_json::Value::Object(cm) = &chunk.metadata {
        for (k, v) in cm {
            merged.insert(k.clone(), v.clone());
        }
    }
    merged.insert("document_id".into(), json!(document.document_id));
    if let Some(source) = &document.source_id {
        merged.insert("source_id".into(), json!(source));
    }
    if let Some(title) = &document.title {
        merged.entry("title".to_string()).or_insert(json!(title));
    }
    serde_json::Value::Object(merged)
}

fn rebuild_ivf(state: &mut State) -> Result<()> {
    if !state.config.index.starts_with("ivf") {
        return Ok(());
    }
    if state.arena.len() < MIN_TRAIN_ROWS {
        state.ivf = None; // too small: search falls back to flat, delta-free
        return Ok(());
    }
    let mut ivf_config = state.config.ivf.clone();
    if state.config.index == "ivf_pq" && ivf_config.pq_m == 0 {
        // auto pq_m: largest divisor of dim with sub_dim >= 4, capped at 64.
        let dim = state.config.dim;
        ivf_config.pq_m = (1..=64.min(dim / 4))
            .rev()
            .find(|m| dim.is_multiple_of(*m))
            .unwrap_or(1);
    } else if state.config.index == "ivf_flat" {
        ivf_config.pq_m = 0;
    }
    state.ivf = Some(IvfIndex::build(&state.arena, &ivf_config)?);
    Ok(())
}

impl VaultEngine {
    /// Open (or create) a vault at `path`. `config` is required on create;
    /// on reopen a differing dimension/metric is rejected.
    pub fn open(path: &Path, config: EngineConfig) -> Result<VaultEngine> {
        if config.dim == 0 {
            return Err(Error::invalid("dim", "a positive dimension", "0"));
        }
        fs::create_dir_all(path)
            .map_err(|e| Error::io(format!("create vault dir {}", path.display()), e))?;

        // Writer lock: one engine per directory.
        let lock_path = path.join("LOCK");
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| Error::io(format!("open lock {}", lock_path.display()), e))?;
        lock_file.try_lock_exclusive().map_err(|_| Error::Locked {
            path: path.display().to_string(),
            owner: "another process".to_string(),
        })?;

        let manifest = snapshot::load_manifest(path)?;
        let mut state = if let Some(manifest) = manifest {
            let mut stored_config: EngineConfig = serde_json::from_value(manifest.config.clone())?;
            // Runtime knobs, not data-format properties: honor the values
            // requested for this open.
            stored_config.storage = config.storage.clone();
            stored_config.ivf.nprobe = config.ivf.nprobe;
            stored_config.flat_threshold = config.flat_threshold;
            if stored_config.dim != config.dim || stored_config.metric != config.metric {
                return Err(Error::invalid(
                    "config",
                    format!(
                        "dim={} metric={:?} (as stored)",
                        stored_config.dim, stored_config.metric
                    ),
                    format!("dim={} metric={:?}", config.dim, config.metric),
                ));
            }
            let arena;
            let persisted;
            if stored_config.storage == "mmap" {
                // load_state verifies the vectors.bin checksum; the parsed
                // Vec<f32> is dropped and the file is served via mmap.
                let (state_only, _vectors) = snapshot::load_state(path, &manifest)?;
                persisted = state_only;
                let vectors_file =
                    fs::File::open(snapshot::vectors_path(path, manifest.generation))
                        .map_err(|e| Error::io("open vectors.bin for mmap", e))?;
                // SAFETY note: memmap2's Mmap::map is unsafe at the API level
                // because the file could be mutated externally; RagVault
                // snapshots are immutable by design (new generations are new
                // files) and the writer lock excludes concurrent writers.
                let mmap = unsafe { memmap2::Mmap::map(&vectors_file) }
                    .map_err(|e| Error::io("mmap vectors.bin", e))?;
                arena = VectorArena::from_mmap(
                    stored_config.dim,
                    stored_config.metric,
                    mmap,
                    persisted.deleted.clone(),
                )?;
            } else {
                let (state_only, vectors) = snapshot::load_state(path, &manifest)?;
                persisted = state_only;
                arena = VectorArena::from_parts(
                    stored_config.dim,
                    stored_config.metric,
                    vectors,
                    persisted.deleted.clone(),
                )?;
            }
            let chunks: Vec<Option<StoredChunk>> = persisted
                .chunks
                .into_iter()
                .map(|c| {
                    c.map(|chunk| {
                        let doc = persisted
                            .documents
                            .iter()
                            .find(|d| d.document_id == chunk.document_id);
                        let eff = doc
                            .map(|d| effective_metadata(d, &chunk))
                            .unwrap_or_else(|| chunk.metadata.clone());
                        StoredChunk {
                            chunk,
                            eff_metadata: eff,
                        }
                    })
                })
                .collect();
            let mut chunk_ids = HashMap::new();
            let mut doc_rows: HashMap<String, Vec<u32>> = HashMap::new();
            for (row, slot) in chunks.iter().enumerate() {
                if let Some(sc) = slot {
                    if !arena.is_deleted(row as u32) {
                        chunk_ids.insert(sc.chunk.chunk_id.clone(), row as u32);
                        doc_rows
                            .entry(sc.chunk.document_id.clone())
                            .or_default()
                            .push(row as u32);
                    }
                }
            }
            let wal = Wal::open(path, stored_config.wal_sync)?;
            let sq8 = if stored_config.quantization == "sq8" {
                let mut q = Sq8Arena::new(stored_config.dim, stored_config.metric)?;
                for row in 0..arena.len() as u32 {
                    q.push(arena.get(row))?;
                }
                Some(q)
            } else {
                None
            };
            State {
                config: stored_config,
                documents: persisted
                    .documents
                    .into_iter()
                    .map(|d| (d.document_id.clone(), d))
                    .collect(),
                versions: persisted.versions,
                chunks,
                chunk_ids,
                doc_rows,
                arena,
                sq8,
                ivf: None,
                hnsw: persisted.hnsw,
                bm25: persisted.bm25,
                sparse: persisted.sparse,
                seq: manifest.seq,
                wal,
                dirty: false,
                generation: manifest.generation,
            }
        } else {
            let wal = Wal::open(path, config.wal_sync)?;
            State {
                arena: VectorArena::new(config.dim, config.metric),
                sq8: if config.quantization == "sq8" {
                    Some(Sq8Arena::new(config.dim, config.metric)?)
                } else {
                    None
                },
                ivf: None,
                hnsw: Hnsw::new(config.hnsw.clone()),
                bm25: Bm25Index::new(config.bm25.clone()),
                sparse: SparseIndex::new(),
                config,
                documents: HashMap::new(),
                versions: HashMap::new(),
                chunks: Vec::new(),
                chunk_ids: HashMap::new(),
                doc_rows: HashMap::new(),
                seq: 0,
                wal,
                dirty: false,
                generation: 0,
            }
        };

        // Recovery: replay WAL operations newer than the snapshot.
        let records = Wal::replay(path, state.seq)?;
        for record in records {
            let seq = record.seq;
            match record.op {
                WalOp::UpsertDocument {
                    document,
                    chunks,
                    dim,
                    sparse,
                } => {
                    Self::apply_upsert_with_sparse(
                        &mut state,
                        document,
                        chunks,
                        &record.payload,
                        dim,
                        sparse,
                    )?;
                }
                WalOp::DeleteDocument { document_id } => {
                    Self::apply_delete(&mut state, &document_id);
                }
            }
            state.seq = seq;
            state.dirty = true;
        }
        rebuild_ivf(&mut state)?;

        Ok(VaultEngine {
            path: path.to_path_buf(),
            state: RwLock::new(state),
            _lock_file: lock_file,
        })
    }

    /// Upsert (insert or atomically replace) a document with its chunks and
    /// dense vectors (`vectors.len() == chunks.len() * dim`, may be empty
    /// alongside empty chunks). Sparse vectors are optional per chunk.
    pub fn upsert_document(
        &self,
        mut document: Document,
        chunks: Vec<Chunk>,
        vectors: &[f32],
        sparse: Option<Vec<Option<SparseVector>>>,
    ) -> Result<u64> {
        let mut state = self.state.write();
        let dim = state.config.dim;
        if vectors.len() != chunks.len() * dim {
            return Err(Error::invalid(
                "vectors",
                format!(
                    "{} f32 values ({} chunks x dim {})",
                    chunks.len() * dim,
                    chunks.len(),
                    dim
                ),
                format!("{}", vectors.len()),
            ));
        }
        if let Some(sv) = &sparse {
            if sv.len() != chunks.len() {
                return Err(Error::invalid(
                    "sparse",
                    format!("{} entries", chunks.len()),
                    format!("{}", sv.len()),
                ));
            }
        }
        // Version bookkeeping: bump over any previous version.
        let next_version = state
            .documents
            .get(&document.document_id)
            .map(|d| d.current_version + 1)
            .unwrap_or(1)
            .max(document.current_version);
        document.current_version = next_version;
        let mut chunks = chunks;
        for c in &mut chunks {
            c.document_id = document.document_id.clone();
            c.document_version = next_version;
        }

        let seq = state.seq + 1;
        let op = WalOp::UpsertDocument {
            document: document.clone(),
            chunks: chunks.clone(),
            dim,
            sparse: sparse.clone(),
        };
        state.wal.append(seq, &op, vectors)?;
        Self::apply_upsert_with_sparse(&mut state, document, chunks, vectors, dim, sparse)?;
        state.seq = seq;
        state.dirty = true;
        Ok(next_version)
    }

    fn apply_upsert_with_sparse(
        state: &mut State,
        document: Document,
        chunks: Vec<Chunk>,
        vectors: &[f32],
        dim: usize,
        sparse: Option<Vec<Option<SparseVector>>>,
    ) -> Result<()> {
        if dim != state.config.dim {
            return Err(Error::DimensionMismatch {
                expected: state.config.dim,
                got: dim,
            });
        }
        // Stage new rows first; only after everything is inserted do we
        // retire the old version — a query in between still sees the old
        // version because we hold the write lock for the whole apply.
        let doc_id = document.document_id.clone();
        let old_rows = state.doc_rows.remove(&doc_id).unwrap_or_default();

        let mut new_rows = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.into_iter().enumerate() {
            let vector = &vectors[i * dim..(i + 1) * dim];
            let row = state.arena.push(vector)?;
            if let Some(sq8) = state.sq8.as_mut() {
                // sq8 mode replaces the graph: quantized scan + f32 rescore.
                sq8.push(state.arena.get(row))?;
            } else if state.config.index.starts_with("ivf") {
                // ivf mode: fresh rows are delta-scanned until the next
                // rebuild (open/flush/compact); no graph maintained.
            } else {
                state.hnsw.insert(&state.arena, row);
            }
            state.bm25.add(row, &chunk.text);
            match sparse.as_ref().and_then(|s| s.get(i).cloned().flatten()) {
                Some(sv) => state.sparse.add(row, &sv)?,
                None => state.sparse.add_empty(row),
            }
            let eff = effective_metadata(&document, &chunk);
            state.chunk_ids.insert(chunk.chunk_id.clone(), row);
            state.chunks.push(Some(StoredChunk {
                chunk,
                eff_metadata: eff,
            }));
            new_rows.push(row);
        }

        // Publish: register document + rows, then tombstone the old rows.
        state
            .versions
            .entry(doc_id.clone())
            .or_default()
            .push(PersistedVersionV1 {
                version: document.current_version,
                content_hash: document
                    .metadata
                    .get("content_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                created_at: now_millis(),
            });
        state.doc_rows.insert(doc_id.clone(), new_rows);
        state.documents.insert(doc_id, document);
        for row in old_rows {
            Self::retire_row(state, row);
        }
        Ok(())
    }

    fn retire_row(state: &mut State, row: u32) {
        if let Some(Some(sc)) = state.chunks.get(row as usize) {
            let cid = sc.chunk.chunk_id.clone();
            if state.chunk_ids.get(&cid) == Some(&row) {
                state.chunk_ids.remove(&cid);
            }
        }
        state.arena.delete(row);
        state.bm25.remove(row);
    }

    pub fn delete_document(&self, document_id: &str) -> Result<bool> {
        let mut state = self.state.write();
        if !state.documents.contains_key(document_id) {
            return Ok(false);
        }
        let seq = state.seq + 1;
        let op = WalOp::DeleteDocument {
            document_id: document_id.to_string(),
        };
        state.wal.append(seq, &op, &[])?;
        Self::apply_delete(&mut state, document_id);
        state.seq = seq;
        state.dirty = true;
        Ok(true)
    }

    fn apply_delete(state: &mut State, document_id: &str) {
        state.documents.remove(document_id);
        if let Some(rows) = state.doc_rows.remove(document_id) {
            for row in rows {
                Self::retire_row(state, row);
            }
        }
    }

    /// Search across signals; hybrid fuses dense + BM25 (+ sparse when a
    /// sparse query is provided) with weighted RRF.
    pub fn search(&self, request: &SearchRequest) -> Result<SearchResponse> {
        let state = self.state.read();
        let k = request.k;
        if k == 0 {
            return Ok(SearchResponse {
                hits: vec![],
                plan: json!({"backend": "none", "reason": ["k = 0"]}),
            });
        }
        let filter = match &request.filter {
            Some(f) => Filter::parse(f)?,
            None => Filter::True,
        };
        let has_filter = !filter.is_true();
        let accept = |row: u32| -> bool {
            if !has_filter {
                return true;
            }
            match state.chunks.get(row as usize) {
                Some(Some(sc)) => filter.matches(&sc.eff_metadata),
                _ => false,
            }
        };

        let pool = request.candidates.unwrap_or((4 * k).max(50));
        let mode = if request.mode == "auto" {
            match (&request.vector, &request.text) {
                (Some(_), Some(_)) => "hybrid",
                (Some(_), None) => "dense",
                (None, Some(_)) => "keyword",
                (None, None) => {
                    if request.sparse.is_some() {
                        "sparse"
                    } else {
                        return Err(Error::invalid(
                            "search request",
                            "a dense vector, query text or sparse vector",
                            "none of them",
                        ));
                    }
                }
            }
        } else {
            request.mode.as_str()
        };

        let mut plan_reasons: Vec<String> = vec![format!("mode = {mode}")];
        let mut dense_backend = "none";

        // -- dense signal --------------------------------------------------
        let mut dense: Vec<(u32, f32)> = Vec::new();
        if matches!(mode, "dense" | "hybrid") {
            let vector = request.vector.as_ref().ok_or_else(|| {
                Error::invalid(
                    "vector",
                    "a dense query vector for dense/hybrid mode",
                    "none",
                )
            })?;
            let prepared = state.arena.prepare_query(vector)?;
            let live = state.arena.live();
            if state.config.index.starts_with("ivf") {
                if let Some(ivf) = state.ivf.as_ref() {
                    dense_backend = if ivf.uses_pq() { "ivf_pq" } else { "ivf_flat" };
                    let nprobe = request.nprobe.unwrap_or(state.config.ivf.nprobe);
                    plan_reasons.push(format!(
                        "ivf: nlist {}, nprobe {nprobe}, delta rows {}",
                        ivf.nlist(),
                        state.arena.len() as u32 - ivf.built_rows()
                    ));
                    dense = ivf.search(&state.arena, &prepared, pool, nprobe, &accept);
                } else {
                    dense_backend = "flat";
                    plan_reasons.push(format!(
                        "ivf configured but only {live} rows (< {MIN_TRAIN_ROWS} to train) — exact flat scan"
                    ));
                    dense = FlatIndex::search(&state.arena, &prepared, pool, &accept);
                }
            } else if let Some(sq8) = state.sq8.as_ref() {
                dense_backend = "sq8_flat";
                let oversample = (pool * 4).max(pool + 16);
                plan_reasons.push(format!(
                    "sq8 quantized scan: oversample {oversample}, f32 rescore to pool {pool}"
                ));
                let quantized = sq8.prepare_query(&prepared)?;
                let candidates = sq8.scan(&quantized, oversample, &|id| {
                    !state.arena.is_deleted(id) && accept(id)
                });
                let mut rescored = TopK::new(pool);
                for (id, _) in candidates {
                    rescored.push(id, state.arena.score(id, &prepared));
                }
                dense = rescored.into_sorted();
            } else if live <= state.config.flat_threshold || state.hnsw.is_empty() {
                dense_backend = "flat";
                plan_reasons.push(format!(
                    "flat scan: {live} live vectors <= flat_threshold {}",
                    state.config.flat_threshold
                ));
                dense = FlatIndex::search(&state.arena, &prepared, pool, &accept);
            } else {
                dense_backend = "hnsw";
                let ef = request
                    .ef_search
                    .unwrap_or(state.config.hnsw.ef_search)
                    .max(pool);
                plan_reasons.push(format!("hnsw: ef_search = {ef}"));
                let accept_dyn: Option<&dyn Fn(u32) -> bool> =
                    if has_filter { Some(&accept) } else { None };
                dense = state
                    .hnsw
                    .search(&state.arena, &prepared, pool, ef, accept_dyn);
                if has_filter && dense.len() < pool.min(state.arena.live()) {
                    // Restrictive filter: retry with a wider beam, then fall
                    // back to an exact filtered scan if still starved.
                    let wide_ef = (ef * 4).max(512);
                    plan_reasons.push(format!(
                        "filtered traversal starved ({} hits) — retry ef {wide_ef}",
                        dense.len()
                    ));
                    dense = state
                        .hnsw
                        .search(&state.arena, &prepared, pool, wide_ef, accept_dyn);
                    if dense.len() < k {
                        dense_backend = "flat_filtered_fallback";
                        plan_reasons
                            .push("still starved — exact filtered flat fallback".to_string());
                        dense = FlatIndex::search(&state.arena, &prepared, pool, &accept);
                    }
                }
            }
        }

        // -- lexical signal ------------------------------------------------
        let mut bm25: Vec<(u32, f32)> = Vec::new();
        if matches!(mode, "keyword" | "hybrid") {
            let text = request.text.as_ref().ok_or_else(|| {
                Error::invalid("text", "query text for keyword/hybrid mode", "none")
            })?;
            bm25 = state.bm25.search(text, pool, &|row| {
                !state.arena.is_deleted(row) && accept(row)
            });
        }

        // -- sparse signal -------------------------------------------------
        let mut sparse_hits: Vec<(u32, f32)> = Vec::new();
        if let Some(sq) = &request.sparse {
            sparse_hits = state
                .sparse
                .search(sq, pool, &|row| !state.arena.is_deleted(row) && accept(row))?;
        }

        // -- fusion --------------------------------------------------------
        let weights = request.weights.clone().unwrap_or_default();
        let weight = |name: &str| weights.get(name).copied().unwrap_or(1.0);
        let fused: Vec<(u32, f32)> = match mode {
            "dense" if request.sparse.is_none() => dense.iter().take(k).copied().collect(),
            "keyword" if request.sparse.is_none() => bm25.iter().take(k).copied().collect(),
            "sparse" => sparse_hits.iter().take(k).copied().collect(),
            _ => {
                let mut inputs = Vec::new();
                if !dense.is_empty() {
                    inputs.push(FusionInput {
                        name: "dense",
                        weight: weight("dense"),
                        ranked: &dense,
                    });
                }
                if !bm25.is_empty() {
                    inputs.push(FusionInput {
                        name: "bm25",
                        weight: weight("bm25"),
                        ranked: &bm25,
                    });
                }
                if !sparse_hits.is_empty() {
                    inputs.push(FusionInput {
                        name: "sparse",
                        weight: weight("sparse"),
                        ranked: &sparse_hits,
                    });
                }
                plan_reasons.push(format!(
                    "weighted RRF over {} signal(s), pool {pool}",
                    inputs.len()
                ));
                rrf_fuse(&inputs, 60.0, k)
            }
        };

        let dense_map: HashMap<u32, f32> = dense.iter().copied().collect();
        let bm25_map: HashMap<u32, f32> = bm25.iter().copied().collect();
        let sparse_map: HashMap<u32, f32> = sparse_hits.iter().copied().collect();
        let hits = fused
            .into_iter()
            .filter_map(|(row, score)| {
                state.chunks.get(row as usize).and_then(|slot| {
                    slot.as_ref().map(|sc| SearchHit {
                        chunk_id: sc.chunk.chunk_id.clone(),
                        document_id: sc.chunk.document_id.clone(),
                        score,
                        dense_score: dense_map.get(&row).copied(),
                        bm25_score: bm25_map.get(&row).copied(),
                        sparse_score: sparse_map.get(&row).copied(),
                        internal_id: row,
                    })
                })
            })
            .collect();

        Ok(SearchResponse {
            hits,
            plan: json!({
                "mode": mode,
                "dense_backend": dense_backend,
                "reason": plan_reasons,
                "candidate_pool": pool,
                "filtered": has_filter,
                "live_vectors": state.arena.live(),
            }),
        })
    }

    pub fn get_chunk(&self, chunk_id: &str) -> Option<Chunk> {
        let state = self.state.read();
        state
            .chunk_ids
            .get(chunk_id)
            .and_then(|&row| state.chunks.get(row as usize))
            .and_then(|slot| slot.as_ref().map(|sc| sc.chunk.clone()))
    }

    pub fn get_document(&self, document_id: &str) -> Option<Document> {
        self.state.read().documents.get(document_id).cloned()
    }

    /// Chunks of the current version, ordered by chunk_index.
    pub fn get_document_chunks(&self, document_id: &str) -> Vec<Chunk> {
        let state = self.state.read();
        let mut chunks: Vec<Chunk> = state
            .doc_rows
            .get(document_id)
            .map(|rows| {
                rows.iter()
                    .filter_map(|&row| {
                        state
                            .chunks
                            .get(row as usize)
                            .and_then(|s| s.as_ref().map(|sc| sc.chunk.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        chunks.sort_by_key(|c| c.chunk_index);
        chunks
    }

    /// Export live rows: (chunk_ids, row-major f32 vectors). Used by the
    /// faiss interop and GPU sidecar builds.
    pub fn export_dense(&self) -> (Vec<String>, Vec<f32>) {
        let state = self.state.read();
        let dim = state.config.dim;
        let mut ids = Vec::new();
        let mut vectors = Vec::new();
        for row in 0..state.arena.len() as u32 {
            if state.arena.is_deleted(row) {
                continue;
            }
            if let Some(Some(sc)) = state.chunks.get(row as usize) {
                ids.push(sc.chunk.chunk_id.clone());
                vectors.extend_from_slice(state.arena.get(row));
            }
        }
        let _ = dim;
        (ids, vectors)
    }

    /// Evaluate a filter against chunks by id (used by sidecar searchers,
    /// e.g. the GPU path, to post-filter with the same DSL semantics).
    pub fn filter_chunks(
        &self,
        chunk_ids: &[String],
        filter: &serde_json::Value,
    ) -> Result<Vec<bool>> {
        let parsed = Filter::parse(filter)?;
        let state = self.state.read();
        Ok(chunk_ids
            .iter()
            .map(|cid| {
                state
                    .chunk_ids
                    .get(cid)
                    .and_then(|&row| state.chunks.get(row as usize))
                    .and_then(|slot| slot.as_ref())
                    .map(|sc| parsed.matches(&sc.eff_metadata))
                    .unwrap_or(false)
            })
            .collect())
    }

    pub fn list_documents(&self) -> Vec<Document> {
        let mut docs: Vec<Document> = self.state.read().documents.values().cloned().collect();
        docs.sort_by(|a, b| a.document_id.cmp(&b.document_id));
        docs
    }

    pub fn list_document_versions(&self, document_id: &str) -> Vec<(u64, String, u64)> {
        self.state
            .read()
            .versions
            .get(document_id)
            .map(|vs| {
                vs.iter()
                    .map(|v| (v.version, v.content_hash.clone(), v.created_at))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Persist a snapshot generation and truncate the WAL. Also retrains
    /// the IVF acceleration structure when configured (it is not persisted;
    /// reopen rebuilds it).
    pub fn flush(&self) -> Result<()> {
        let mut state = self.state.write();
        if !state.dirty {
            return Ok(());
        }
        rebuild_ivf(&mut state)?;
        state.wal.sync()?;
        let generation = state.generation + 1;
        let persisted = PersistedStateV1 {
            documents: state.documents.values().cloned().collect(),
            versions: state.versions.clone(),
            chunks: state
                .chunks
                .iter()
                .map(|s| s.as_ref().map(|sc| sc.chunk.clone()))
                .collect(),
            deleted: state.arena.deleted_bitmap().to_vec(),
            bm25: state.bm25.clone(),
            hnsw: state.hnsw.clone(),
            sparse: state.sparse.clone(),
        };
        let (base_part, tail_part) = state.arena.vector_parts();
        snapshot::publish(
            &self.path,
            generation,
            state.seq,
            serde_json::to_value(&state.config)?,
            &persisted,
            &[base_part, tail_part],
        )?;
        state.wal.truncate()?;
        state.generation = generation;
        state.dirty = false;
        Ok(())
    }

    /// Rebuild arenas and indexes without tombstones, then flush.
    /// Synchronous and deterministic (v0.1 — background compaction is
    /// planned).
    pub fn compact(&self) -> Result<()> {
        {
            let mut state = self.state.write();
            let config = state.config.clone();
            let mut arena = VectorArena::new(config.dim, config.metric);
            let mut hnsw = Hnsw::new(config.hnsw.clone());
            let mut bm25 = Bm25Index::new(config.bm25.clone());
            let mut sq8 = if state.sq8.is_some() {
                Some(Sq8Arena::new(config.dim, config.metric)?)
            } else {
                None
            };
            let mut chunks: Vec<Option<StoredChunk>> = Vec::new();
            let mut chunk_ids = HashMap::new();
            let mut doc_rows: HashMap<String, Vec<u32>> = HashMap::new();
            let mut row_map: HashMap<u32, u32> = HashMap::new();

            let mut live_rows: Vec<u32> = (0..state.chunks.len() as u32)
                .filter(|&r| !state.arena.is_deleted(r) && state.chunks[r as usize].is_some())
                .collect();
            live_rows.sort();
            for old_row in live_rows {
                let sc = state.chunks[old_row as usize].as_ref().expect("live row");
                let vector = state.arena.get(old_row).to_vec();
                let new_row = arena.push(&vector)?;
                if let Some(q) = sq8.as_mut() {
                    q.push(arena.get(new_row))?;
                } else {
                    hnsw.insert(&arena, new_row);
                }
                bm25.add(new_row, &sc.chunk.text);
                row_map.insert(old_row, new_row);
                chunk_ids.insert(sc.chunk.chunk_id.clone(), new_row);
                doc_rows
                    .entry(sc.chunk.document_id.clone())
                    .or_default()
                    .push(new_row);
                chunks.push(Some(StoredChunk {
                    chunk: sc.chunk.clone(),
                    eff_metadata: sc.eff_metadata.clone(),
                }));
            }
            state.sparse = state.sparse.remap(&row_map, arena.len() as u32);
            state.arena = arena;
            state.sq8 = sq8;
            state.hnsw = hnsw;
            state.bm25 = bm25;
            state.chunks = chunks;
            state.chunk_ids = chunk_ids;
            state.doc_rows = doc_rows;
            rebuild_ivf(&mut state)?;
            state.dirty = true;
        }
        self.flush()
    }

    pub fn stats(&self) -> serde_json::Value {
        let state = self.state.read();
        json!({
            "documents": state.documents.len(),
            "live_chunks": state.arena.live(),
            "total_rows": state.arena.len(),
            "tombstones": state.arena.len() - state.arena.live(),
            "dim": state.config.dim,
            "metric": state.config.metric,
            "bm25_terms": state.bm25.term_count(),
            "generation": state.generation,
            "seq": state.seq,
            "dirty": state.dirty,
            "hnsw_nodes": state.hnsw.len(),
            "quantization": state.config.quantization,
            "sq8_bytes": state.sq8.as_ref().map(|q| q.memory_bytes()),
            "index": state.config.index,
            "storage": if state.arena.is_mmap() { "mmap" } else { "memory" },
            "ivf": state.ivf.as_ref().map(|ivf| json!({
                "nlist": ivf.nlist(),
                "pq": ivf.uses_pq(),
                "built_rows": ivf.built_rows(),
                "memory_bytes": ivf.memory_bytes(),
            })),
        })
    }

    pub fn config(&self) -> EngineConfig {
        self.state.read().config.clone()
    }

    pub fn close(&self) -> Result<()> {
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(dim: usize) -> EngineConfig {
        let mut c = EngineConfig::new(dim);
        c.wal_sync = SyncPolicy::Sync;
        c.flat_threshold = 10; // exercise hnsw path early in tests
        c
    }

    fn doc(id: &str, metadata: serde_json::Value) -> Document {
        Document {
            document_id: id.into(),
            source_id: None,
            current_version: 1,
            title: Some(format!("Title of {id}")),
            metadata,
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

    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[hot % dim] = 1.0;
        v
    }

    #[test]
    fn upsert_search_delete_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();

        engine
            .upsert_document(
                doc("a", json!({"lang": "en"})),
                vec![chunk("a", 0, "cancellation policy refunds")],
                &unit_vec(4, 0),
                None,
            )
            .unwrap();
        engine
            .upsert_document(
                doc("b", json!({"lang": "pt"})),
                vec![chunk("b", 0, "shipping and delivery")],
                &unit_vec(4, 1),
                None,
            )
            .unwrap();

        let response = engine
            .search(&SearchRequest {
                vector: Some(unit_vec(4, 0)),
                text: Some("cancellation".into()),
                sparse: None,
                k: 2,
                mode: "hybrid".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap();
        assert_eq!(response.hits[0].document_id, "a");
        assert!(response.hits[0].dense_score.is_some());
        assert!(response.hits[0].bm25_score.is_some());

        assert!(engine.delete_document("a").unwrap());
        assert!(!engine.delete_document("a").unwrap());
        let response = engine
            .search(&SearchRequest {
                vector: Some(unit_vec(4, 0)),
                text: None,
                sparse: None,
                k: 5,
                mode: "dense".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap();
        assert!(response.hits.iter().all(|h| h.document_id != "a"));
    }

    #[test]
    fn replace_publishes_atomically_and_hides_old_version() {
        let dir = tempfile::tempdir().unwrap();
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        engine
            .upsert_document(
                doc("a", json!({})),
                vec![chunk("a", 0, "old content about zebras")],
                &unit_vec(4, 0),
                None,
            )
            .unwrap();
        let v2 = engine
            .upsert_document(
                doc("a", json!({})),
                vec![chunk("a", 0, "new content about lions")],
                &unit_vec(4, 1),
                None,
            )
            .unwrap();
        assert_eq!(v2, 2);
        let hits = engine
            .search(&SearchRequest {
                vector: None,
                text: Some("zebras".into()),
                sparse: None,
                k: 5,
                mode: "keyword".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap()
            .hits;
        assert!(hits.is_empty(), "old version must be invisible");
        let chunks = engine.get_document_chunks("a");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].document_version, 2);
        assert_eq!(engine.list_document_versions("a").len(), 2);
    }

    #[test]
    fn filters_apply_to_all_signals() {
        let dir = tempfile::tempdir().unwrap();
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        for (id, lang, hot) in [("a", "en", 0), ("b", "pt", 0), ("c", "en", 1)] {
            engine
                .upsert_document(
                    doc(id, json!({"lang": lang})),
                    vec![chunk(id, 0, "shared term document")],
                    &unit_vec(4, hot),
                    None,
                )
                .unwrap();
        }
        let request = SearchRequest {
            vector: Some(unit_vec(4, 0)),
            text: Some("shared".into()),
            sparse: None,
            k: 10,
            mode: "hybrid".into(),
            candidates: None,
            filter: Some(json!({"lang": "en"})),
            ef_search: None,
            nprobe: None,
            weights: None,
        };
        let hits = engine.search(&request).unwrap().hits;
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| h.document_id != "b"));
    }

    #[test]
    fn reopen_recovers_from_wal_without_flush() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            engine
                .upsert_document(
                    doc("a", json!({})),
                    vec![chunk("a", 0, "durable content")],
                    &unit_vec(4, 0),
                    None,
                )
                .unwrap();
            // no flush — drop simulates a crash after WAL write
        }
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        let hits = engine
            .search(&SearchRequest {
                vector: None,
                text: Some("durable".into()),
                sparse: None,
                k: 5,
                mode: "keyword".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap()
            .hits;
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn reopen_after_flush_uses_snapshot_plus_wal() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            engine
                .upsert_document(
                    doc("a", json!({})),
                    vec![chunk("a", 0, "snapshotted")],
                    &unit_vec(4, 0),
                    None,
                )
                .unwrap();
            engine.flush().unwrap();
            engine
                .upsert_document(
                    doc("b", json!({})),
                    vec![chunk("b", 0, "only in wal")],
                    &unit_vec(4, 1),
                    None,
                )
                .unwrap();
        }
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        assert_eq!(engine.list_documents().len(), 2);
        for term in ["snapshotted", "wal"] {
            let hits = engine
                .search(&SearchRequest {
                    vector: None,
                    text: Some(term.into()),
                    sparse: None,
                    k: 5,
                    mode: "keyword".into(),
                    candidates: None,
                    filter: None,
                    ef_search: None,
                    nprobe: None,
                    weights: None,
                })
                .unwrap()
                .hits;
            assert_eq!(hits.len(), 1, "term {term}");
        }
    }

    #[test]
    fn double_reopen_replay_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            engine
                .upsert_document(
                    doc("a", json!({})),
                    vec![chunk("a", 0, "idempotent replay")],
                    &unit_vec(4, 0),
                    None,
                )
                .unwrap();
        }
        for _ in 0..2 {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            assert_eq!(engine.list_documents().len(), 1);
            assert_eq!(engine.get_document_chunks("a").len(), 1);
        }
    }

    #[test]
    fn compact_drops_tombstones_and_preserves_results() {
        let dir = tempfile::tempdir().unwrap();
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        for i in 0..20 {
            let id = format!("d{i}");
            engine
                .upsert_document(
                    doc(&id, json!({})),
                    vec![chunk(&id, 0, &format!("document number {i}"))],
                    &unit_vec(4, i),
                    None,
                )
                .unwrap();
        }
        for i in 0..10 {
            engine.delete_document(&format!("d{i}")).unwrap();
        }
        let before = engine
            .search(&SearchRequest {
                vector: Some(unit_vec(4, 3)),
                text: None,
                sparse: None,
                k: 5,
                mode: "dense".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap()
            .hits
            .iter()
            .map(|h| h.document_id.clone())
            .collect::<Vec<_>>();
        engine.compact().unwrap();
        let stats = engine.stats();
        assert_eq!(stats["tombstones"], 0);
        assert_eq!(stats["documents"], 10);
        let after = engine
            .search(&SearchRequest {
                vector: Some(unit_vec(4, 3)),
                text: None,
                sparse: None,
                k: 5,
                mode: "dense".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap()
            .hits
            .iter()
            .map(|h| h.document_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(before, after);
    }

    #[test]
    fn second_writer_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let _first = VaultEngine::open(dir.path(), config(4)).unwrap();
        let second = VaultEngine::open(dir.path(), config(4));
        assert!(matches!(second, Err(Error::Locked { .. })));
    }

    #[test]
    fn dimension_mismatch_on_reopen_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            engine
                .upsert_document(
                    doc("a", json!({})),
                    vec![chunk("a", 0, "x")],
                    &unit_vec(4, 0),
                    None,
                )
                .unwrap();
            engine.flush().unwrap();
        }
        let wrong = VaultEngine::open(dir.path(), config(8));
        assert!(wrong.is_err());
    }

    #[test]
    fn sq8_backend_full_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(8);
        cfg.quantization = "sq8".to_string();
        let dense_req = |hot: usize| SearchRequest {
            vector: Some(unit_vec(8, hot)),
            text: None,
            sparse: None,
            k: 3,
            mode: "dense".into(),
            candidates: None,
            filter: None,
            ef_search: None,
            nprobe: None,
            weights: None,
        };
        {
            let engine = VaultEngine::open(dir.path(), cfg.clone()).unwrap();
            for i in 0..30 {
                let id = format!("d{i}");
                engine
                    .upsert_document(
                        doc(&id, json!({"parity": i % 2})),
                        vec![chunk(&id, 0, &format!("doc {i}"))],
                        &unit_vec(8, i),
                        None,
                    )
                    .unwrap();
            }
            let response = engine.search(&dense_req(3)).unwrap();
            assert_eq!(response.plan["dense_backend"], "sq8_flat");
            assert_eq!(response.hits[0].document_id, "d3");
            assert_eq!(engine.stats()["hnsw_nodes"], 0, "sq8 mode skips the graph");
            assert!(engine.stats()["sq8_bytes"].as_u64().unwrap() > 0);

            // filters are a true prefilter on the quantized scan
            let mut filtered = dense_req(3);
            filtered.filter = Some(json!({"parity": 0}));
            let hits = engine.search(&filtered).unwrap().hits;
            assert!(!hits.is_empty());
            assert!(hits
                .iter()
                .all(|h| h.document_id[1..].parse::<usize>().unwrap() % 2 == 0));
            engine.flush().unwrap();
        }
        // reopen rebuilds the quantized arena from the snapshot
        let engine = VaultEngine::open(dir.path(), cfg.clone()).unwrap();
        let response = engine.search(&dense_req(5)).unwrap();
        assert_eq!(response.plan["dense_backend"], "sq8_flat");
        assert_eq!(response.hits[0].document_id, "d5");
        // compaction preserves the quantized backend
        for i in 0..10 {
            engine.delete_document(&format!("d{i}")).unwrap();
        }
        engine.compact().unwrap();
        let response = engine.search(&dense_req(15)).unwrap();
        assert_eq!(response.plan["dense_backend"], "sq8_flat");
        assert_eq!(response.hits[0].document_id, "d15");
    }

    #[test]
    fn ivf_backend_full_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(8);
        cfg.index = "ivf_flat".to_string();
        cfg.ivf.nprobe = 64; // small collection: probe everything
        let dense_req = |hot: usize| SearchRequest {
            vector: Some(unit_vec(8, hot)),
            text: None,
            sparse: None,
            k: 3,
            mode: "dense".into(),
            candidates: None,
            filter: None,
            ef_search: None,
            nprobe: None,
            weights: None,
        };
        {
            let engine = VaultEngine::open(dir.path(), cfg.clone()).unwrap();
            // Below MIN_TRAIN_ROWS: planner reports the exact-flat fallback.
            for i in 0..40 {
                let id = format!("d{i}");
                engine
                    .upsert_document(
                        doc(&id, json!({})),
                        vec![chunk(&id, 0, &format!("doc {i}"))],
                        &unit_vec(8, i),
                        None,
                    )
                    .unwrap();
            }
            let response = engine.search(&dense_req(3)).unwrap();
            assert_eq!(response.plan["dense_backend"], "flat");
            assert_eq!(response.hits[0].document_id, "d3");

            // Cross the training threshold, then flush() trains the IVF.
            for i in 40..300 {
                let id = format!("d{i}");
                engine
                    .upsert_document(
                        doc(&id, json!({})),
                        vec![chunk(&id, 0, &format!("doc {i}"))],
                        &unit_vec(8, i),
                        None,
                    )
                    .unwrap();
            }
            engine.flush().unwrap();
            let response = engine.search(&dense_req(5)).unwrap();
            assert_eq!(response.plan["dense_backend"], "ivf_flat");
            // per-request nprobe override is honored and visible in the plan
            let mut one_probe = dense_req(5);
            one_probe.nprobe = Some(1);
            let plan = engine.search(&one_probe).unwrap().plan;
            let reason = plan["reason"].to_string();
            assert!(
                reason.contains("nprobe 1"),
                "plan must show nprobe 1: {reason}"
            );
            assert_eq!(response.hits[0].document_id, "d5");
            assert!(engine.stats()["ivf"]["nlist"].as_u64().unwrap() >= 16);

            // Fresh row after the build is found via the delta scan.
            engine
                .upsert_document(
                    doc("fresh", json!({})),
                    vec![chunk("fresh", 0, "fresh doc")],
                    &[0.5; 8],
                    None,
                )
                .unwrap();
            let mut req = dense_req(0);
            req.vector = Some(vec![0.5; 8]);
            let response = engine.search(&req).unwrap();
            assert_eq!(response.hits[0].document_id, "fresh");
        }
        // Reopen rebuilds the IVF from snapshot + WAL.
        let engine = VaultEngine::open(dir.path(), cfg.clone()).unwrap();
        let response = engine.search(&dense_req(7)).unwrap();
        assert_eq!(response.plan["dense_backend"], "ivf_flat");
        assert_eq!(response.hits[0].document_id, "d7");
        // Compaction retrains and preserves results (still >= MIN_TRAIN_ROWS).
        for i in 0..30 {
            engine.delete_document(&format!("d{i}")).unwrap();
        }
        engine.compact().unwrap();
        let response = engine.search(&dense_req(100)).unwrap();
        assert_eq!(response.plan["dense_backend"], "ivf_flat");
        // dim-8 unit vectors alias by i % 8: any live doc in the same class
        // is a legitimate exact winner (ties break by internal id).
        let top: usize = response.hits[0].document_id[1..].parse().unwrap();
        assert_eq!(top % 8, 100 % 8, "top hit must share the query vector");
        // Shrinking below the training threshold falls back to exact flat.
        for i in 30..60 {
            engine.delete_document(&format!("d{i}")).unwrap();
        }
        engine.compact().unwrap();
        let response = engine.search(&dense_req(200)).unwrap();
        assert_eq!(response.plan["dense_backend"], "flat");
        let top: usize = response.hits[0].document_id[1..].parse().unwrap();
        assert_eq!(top % 8, 200 % 8);
    }

    #[test]
    fn mmap_storage_full_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let mut mem_cfg = config(4);
        mem_cfg.flat_threshold = 1_000_000; // flat: deterministic comparison
        let mut mmap_cfg = mem_cfg.clone();
        mmap_cfg.storage = "mmap".to_string();
        let req = |hot: usize| SearchRequest {
            vector: Some(unit_vec(4, hot)),
            text: None,
            sparse: None,
            k: 5,
            mode: "dense".into(),
            candidates: None,
            filter: None,
            ef_search: None,
            nprobe: None,
            weights: None,
        };
        {
            let engine = VaultEngine::open(dir.path(), mem_cfg.clone()).unwrap();
            for i in 0..50 {
                let id = format!("d{i}");
                engine
                    .upsert_document(
                        doc(&id, json!({})),
                        vec![chunk(&id, 0, &format!("doc {i}"))],
                        &unit_vec(4, i),
                        None,
                    )
                    .unwrap();
            }
            engine.flush().unwrap();
        }
        // Reopen with mmap: identical results to a memory reopen.
        let expected = {
            let engine = VaultEngine::open(dir.path(), mem_cfg.clone()).unwrap();
            engine.search(&req(2)).unwrap().hits
        };
        {
            let engine = VaultEngine::open(dir.path(), mmap_cfg.clone()).unwrap();
            assert_eq!(engine.stats()["storage"], "mmap");
            let hits = engine.search(&req(2)).unwrap().hits;
            assert_eq!(
                hits.iter()
                    .map(|h| (&h.chunk_id, h.score))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|h| (&h.chunk_id, h.score))
                    .collect::<Vec<_>>(),
                "mmap must serve byte-identical vectors"
            );
            // Writes after an mmap open land in the RAM tail and are visible.
            engine
                .upsert_document(
                    doc("tail", json!({})),
                    vec![chunk("tail", 0, "tail doc")],
                    &[0.9, 0.1, 0.0, 0.0],
                    None,
                )
                .unwrap();
            let mut tail_req = req(0);
            tail_req.vector = Some(vec![0.9, 0.1, 0.0, 0.0]);
            let hits = engine.search(&tail_req).unwrap().hits;
            assert_eq!(hits[0].document_id, "tail");
            // Deleting an mmap-resident row tombstones it like any other.
            engine.delete_document("d2").unwrap();
            let hits = engine.search(&req(2)).unwrap().hits;
            assert!(hits.iter().all(|h| h.document_id != "d2"));
            // flush() writes base+tail into the next generation (the old
            // generation file may be unlinked while still mapped — fine on
            // POSIX; documented Windows caveat in docs/STORAGE.md).
            engine.flush().unwrap();
        }
        // Third open (mmap again) sees the merged state.
        let engine = VaultEngine::open(dir.path(), mmap_cfg).unwrap();
        assert_eq!(engine.stats()["documents"], 50); // 50 - d2 + tail
        let hits = engine
            .search(&SearchRequest {
                vector: Some(vec![0.9, 0.1, 0.0, 0.0]),
                text: None,
                sparse: None,
                k: 1,
                mode: "dense".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap()
            .hits;
        assert_eq!(hits[0].document_id, "tail");
    }

    #[test]
    fn ivf_pq_auto_subspaces() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(8);
        cfg.index = "ivf_pq".to_string();
        cfg.ivf.nprobe = 64;
        let engine = VaultEngine::open(dir.path(), cfg).unwrap();
        for i in 0..300 {
            let id = format!("d{i}");
            engine
                .upsert_document(
                    doc(&id, json!({})),
                    vec![chunk(&id, 0, &format!("doc {i}"))],
                    &unit_vec(8, i),
                    None,
                )
                .unwrap();
        }
        engine.flush().unwrap();
        let response = engine
            .search(&SearchRequest {
                vector: Some(unit_vec(8, 6)),
                text: None,
                sparse: None,
                k: 3,
                mode: "dense".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap();
        assert_eq!(response.plan["dense_backend"], "ivf_pq");
        assert_eq!(response.hits[0].document_id, "d6");
        assert_eq!(engine.stats()["ivf"]["pq"], true);
    }

    #[test]
    fn sq8_rejects_l2_metric() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(4);
        cfg.quantization = "sq8".to_string();
        cfg.metric = Metric::L2;
        assert!(VaultEngine::open(dir.path(), cfg).is_err());
    }

    #[test]
    fn sparse_survives_wal_replay_and_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let sv = SparseVector {
            indices: vec![3, 9],
            values: vec![1.5, 2.0],
        };
        let sparse_query = SearchRequest {
            vector: None,
            text: None,
            sparse: Some(sv.clone()),
            k: 5,
            mode: "sparse".into(),
            candidates: None,
            filter: None,
            ef_search: None,
            nprobe: None,
            weights: None,
        };
        {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            engine
                .upsert_document(
                    doc("keep", json!({})),
                    vec![chunk("keep", 0, "kept sparse doc")],
                    &unit_vec(4, 0),
                    Some(vec![Some(sv.clone())]),
                )
                .unwrap();
            engine
                .upsert_document(
                    doc("drop", json!({})),
                    vec![chunk("drop", 0, "dropped doc")],
                    &unit_vec(4, 1),
                    None,
                )
                .unwrap();
            // no flush: crash-simulating drop — sparse must come back via WAL
        }
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        let hits = engine.search(&sparse_query).unwrap().hits;
        assert_eq!(hits.len(), 1, "sparse signal must survive WAL replay");
        assert_eq!(hits[0].document_id, "keep");
        assert_eq!(hits[0].sparse_score, Some(1.5 * 1.5 + 2.0 * 2.0));

        // Compaction must remap sparse postings, not drop them.
        engine.delete_document("drop").unwrap();
        engine.compact().unwrap();
        let hits = engine.search(&sparse_query).unwrap().hits;
        assert_eq!(hits.len(), 1, "sparse signal must survive compaction");
        assert_eq!(hits[0].document_id, "keep");
    }

    #[test]
    fn interrupted_publish_leftovers_do_not_break_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            engine
                .upsert_document(
                    doc("a", json!({})),
                    vec![chunk("a", 0, "survives interrupted publish")],
                    &unit_vec(4, 0),
                    None,
                )
                .unwrap();
            engine.flush().unwrap();
        }
        // Simulate a crash mid-publish: stale tmp generation + tmp manifest.
        std::fs::create_dir_all(dir.path().join("gen-99.tmp")).unwrap();
        std::fs::write(dir.path().join("gen-99.tmp/state.json"), b"partial").unwrap();
        std::fs::write(dir.path().join("manifest.json.tmp"), b"{partial").unwrap();

        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        assert_eq!(engine.list_documents().len(), 1);
        let hits = engine
            .search(&SearchRequest {
                vector: None,
                text: Some("survives".into()),
                sparse: None,
                k: 5,
                mode: "keyword".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap()
            .hits;
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn corrupt_snapshot_is_detected_not_silently_loaded() {
        let dir = tempfile::tempdir().unwrap();
        {
            let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
            engine
                .upsert_document(
                    doc("a", json!({})),
                    vec![chunk("a", 0, "content")],
                    &unit_vec(4, 0),
                    None,
                )
                .unwrap();
            engine.flush().unwrap();
        }
        // Flip a byte in the snapshot state file.
        let gen_dir = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .find(|e| e.file_name().to_string_lossy().starts_with("gen-"))
            .unwrap()
            .path();
        let state_path = gen_dir.join("state.json");
        let mut bytes = std::fs::read(&state_path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&state_path, &bytes).unwrap();

        let result = VaultEngine::open(dir.path(), config(4));
        assert!(
            matches!(result, Err(Error::Corrupt { .. })),
            "corruption must surface as an explicit error"
        );
    }

    #[test]
    fn sparse_signal_participates_in_search() {
        let dir = tempfile::tempdir().unwrap();
        let engine = VaultEngine::open(dir.path(), config(4)).unwrap();
        let sv = SparseVector {
            indices: vec![7],
            values: vec![2.0],
        };
        engine
            .upsert_document(
                doc("a", json!({})),
                vec![chunk("a", 0, "sparse doc")],
                &unit_vec(4, 0),
                Some(vec![Some(sv.clone())]),
            )
            .unwrap();
        engine
            .upsert_document(
                doc("b", json!({})),
                vec![chunk("b", 0, "no sparse")],
                &unit_vec(4, 1),
                None,
            )
            .unwrap();
        let hits = engine
            .search(&SearchRequest {
                vector: None,
                text: None,
                sparse: Some(sv),
                k: 5,
                mode: "sparse".into(),
                candidates: None,
                filter: None,
                ef_search: None,
                nprobe: None,
                weights: None,
            })
            .unwrap()
            .hits;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_id, "a");
    }
}
