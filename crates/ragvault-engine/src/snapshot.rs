//! Snapshot persistence: versioned DTOs, checksummed files, atomic
//! generation publish.
//!
//! Layout inside the vault directory:
//!
//! ```text
//! <vault>/
//! ├── LOCK              (writer lock, fs2 exclusive)
//! ├── wal.log
//! ├── manifest.json     (points at the current generation, atomic rename)
//! └── gen-<N>/
//!     ├── state.json    (docs, chunks, bm25, hnsw graph — versioned DTO)
//!     └── vectors.bin   (raw f32 LE, row-major)
//! ```
//!
//! Publish protocol: write `gen-N.tmp/`, fsync every file, rename the dir,
//! write `manifest.json.tmp`, fsync, rename over `manifest.json`, fsync the
//! directory. The previous generation is removed only after the new
//! manifest is durable, so a crash at any point leaves a readable vault.
//!
//! v0.1 serializes state as JSON DTOs (`format_version = 1`). This is
//! honest about its trade-off: robust and debuggable, but not the fastest
//! reopen for very large vaults. A binary segment format is planned
//! (TASKS.md) behind the same manifest, with format_version gating.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use ragvault_core::{Chunk, Document, Error, Result};
use ragvault_retrieval::Bm25Index;
use ragvault_vector::Hnsw;

pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedManifestV1 {
    pub format_version: u32,
    pub generation: u64,
    /// Highest sequence number included in this snapshot.
    pub seq: u64,
    pub files: HashMap<String, PersistedFileV1>,
    /// Engine configuration JSON (dim, metric, hnsw, bm25 params).
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedFileV1 {
    pub len: u64,
    pub crc32: u32,
}

/// Version history entry for a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVersionV1 {
    pub version: u64,
    pub content_hash: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedStateV1 {
    pub documents: Vec<Document>,
    pub versions: HashMap<String, Vec<PersistedVersionV1>>,
    /// Row-aligned with the vector arena; `None` = tombstoned row.
    pub chunks: Vec<Option<Chunk>>,
    pub deleted: Vec<bool>,
    pub bm25: Bm25Index,
    pub hnsw: Hnsw,
    pub sparse: ragvault_retrieval::SparseIndex,
}

fn crc_of(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn write_file_sync(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f =
        fs::File::create(path).map_err(|e| Error::io(format!("create {}", path.display()), e))?;
    f.write_all(bytes)
        .map_err(|e| Error::io(format!("write {}", path.display()), e))?;
    f.sync_all()
        .map_err(|e| Error::io(format!("fsync {}", path.display()), e))?;
    Ok(())
}

fn fsync_dir(dir: &Path) -> Result<()> {
    // Directory fsync is best-effort on non-POSIX targets.
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("manifest.json")
}

/// Load the current manifest, if a snapshot exists.
pub fn load_manifest(dir: &Path) -> Result<Option<PersistedManifestV1>> {
    let path = manifest_path(dir);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::io(format!("read {}", path.display()), e)),
    };
    let manifest: PersistedManifestV1 = serde_json::from_slice(&bytes)
        .map_err(|e| Error::corrupt(path.display().to_string(), e.to_string()))?;
    if manifest.format_version > FORMAT_VERSION {
        return Err(Error::IncompatibleFormat {
            path: path.display().to_string(),
            found: manifest.format_version,
            supported: FORMAT_VERSION,
        });
    }
    Ok(Some(manifest))
}

/// Write a new snapshot generation atomically and return its manifest.
pub fn publish(
    dir: &Path,
    generation: u64,
    seq: u64,
    config: serde_json::Value,
    state: &PersistedStateV1,
    vector_parts: &[&[f32]],
) -> Result<PersistedManifestV1> {
    let gen_name = format!("gen-{generation}");
    let tmp_dir = dir.join(format!("{gen_name}.tmp"));
    let final_dir = dir.join(&gen_name);
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir)
            .map_err(|e| Error::io(format!("clean {}", tmp_dir.display()), e))?;
    }
    fs::create_dir_all(&tmp_dir)
        .map_err(|e| Error::io(format!("mkdir {}", tmp_dir.display()), e))?;

    let state_bytes = serde_json::to_vec(state)?;
    let vector_bytes: Vec<u8> = vector_parts
        .iter()
        .flat_map(|part| part.iter())
        .flat_map(|f| f.to_le_bytes())
        .collect();

    write_file_sync(&tmp_dir.join("state.json"), &state_bytes)?;
    write_file_sync(&tmp_dir.join("vectors.bin"), &vector_bytes)?;

    let mut files = HashMap::new();
    files.insert(
        format!("{gen_name}/state.json"),
        PersistedFileV1 {
            len: state_bytes.len() as u64,
            crc32: crc_of(&state_bytes),
        },
    );
    files.insert(
        format!("{gen_name}/vectors.bin"),
        PersistedFileV1 {
            len: vector_bytes.len() as u64,
            crc32: crc_of(&vector_bytes),
        },
    );

    if final_dir.exists() {
        fs::remove_dir_all(&final_dir)
            .map_err(|e| Error::io(format!("clean {}", final_dir.display()), e))?;
    }
    fs::rename(&tmp_dir, &final_dir)
        .map_err(|e| Error::io(format!("publish {}", final_dir.display()), e))?;
    fsync_dir(dir)?;

    let manifest = PersistedManifestV1 {
        format_version: FORMAT_VERSION,
        generation,
        seq,
        files,
        config,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let tmp_manifest = dir.join("manifest.json.tmp");
    write_file_sync(&tmp_manifest, &manifest_bytes)?;
    fs::rename(&tmp_manifest, manifest_path(dir)).map_err(|e| Error::io("publish manifest", e))?;
    fsync_dir(dir)?;

    // Garbage-collect older generations only after the manifest is durable.
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if (name.starts_with("gen-") && name != gen_name)
                || name.ends_with(".tmp") && name != "manifest.json.tmp"
            {
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }
    Ok(manifest)
}

/// Path of a generation's vectors file (for mmap-backed opens).
pub fn vectors_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("gen-{generation}/vectors.bin"))
}

/// Verify the vectors file checksum without materializing f32s (used by the
/// mmap open path; reading via the page cache is the verification pass).
pub fn verify_vectors_file(dir: &Path, manifest: &PersistedManifestV1) -> Result<()> {
    let rel = format!("gen-{}/vectors.bin", manifest.generation);
    let meta = manifest
        .files
        .get(&rel)
        .ok_or_else(|| Error::corrupt(rel.clone(), "manifest missing vectors.bin entry"))?;
    let path = dir.join(&rel);
    let bytes = fs::read(&path).map_err(|e| Error::io(format!("read {}", path.display()), e))?;
    if bytes.len() as u64 != meta.len || crc_of(&bytes) != meta.crc32 {
        return Err(Error::corrupt(
            path.display().to_string(),
            format!(
                "checksum/length mismatch (expected {} bytes crc {:#x})",
                meta.len, meta.crc32
            ),
        ));
    }
    Ok(())
}

/// Load the snapshot referenced by a manifest, verifying checksums.
pub fn load_state(
    dir: &Path,
    manifest: &PersistedManifestV1,
) -> Result<(PersistedStateV1, Vec<f32>)> {
    let gen_name = format!("gen-{}", manifest.generation);
    let mut state: Option<PersistedStateV1> = None;
    let mut vectors: Option<Vec<f32>> = None;
    for (rel, meta) in &manifest.files {
        let path = dir.join(rel);
        let bytes =
            fs::read(&path).map_err(|e| Error::io(format!("read {}", path.display()), e))?;
        if bytes.len() as u64 != meta.len || crc_of(&bytes) != meta.crc32 {
            return Err(Error::corrupt(
                path.display().to_string(),
                format!(
                    "checksum/length mismatch (expected {} bytes crc {:#x})",
                    meta.len, meta.crc32
                ),
            ));
        }
        if rel.ends_with("state.json") {
            state = Some(
                serde_json::from_slice(&bytes)
                    .map_err(|e| Error::corrupt(path.display().to_string(), e.to_string()))?,
            );
        } else if rel.ends_with("vectors.bin") {
            vectors = Some(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
                    .collect(),
            );
        }
    }
    let state = state.ok_or_else(|| {
        Error::corrupt(gen_name.clone(), "manifest missing state.json".to_string())
    })?;
    let vectors = vectors
        .ok_or_else(|| Error::corrupt(gen_name, "manifest missing vectors.bin".to_string()))?;
    Ok((state, vectors))
}
