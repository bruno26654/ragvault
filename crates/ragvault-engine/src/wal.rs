//! Write-ahead log.
//!
//! Record framing (little-endian):
//!
//! ```text
//! [u32 header_len][u32 payload_len][u64 seq][header JSON][payload bytes][u32 crc32]
//! ```
//!
//! The CRC covers seq + header + payload. Replay stops at the first
//! incomplete or corrupt record and truncates the tail (torn writes from a
//! crash are expected and safe: the operation was never acknowledged).
//! Vectors ride in the binary payload; the JSON header carries the
//! operation, so records stay debuggable without base64 bloat.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use ragvault_core::{Chunk, Document, Error, Result};

/// Durability policy for WAL writes.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SyncPolicy {
    /// fsync after every commit — maximum durability.
    Sync,
    /// flush to the OS after every commit, fsync on flush()/close() —
    /// survives process crashes, may lose the tail on power loss.
    #[default]
    Batch,
}

/// Logical operations recorded in the WAL header.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOp {
    UpsertDocument {
        document: Document,
        chunks: Vec<Chunk>,
        /// dims of the dense vectors in the payload (payload holds
        /// `chunks.len() * dim` f32 LE values; 0 = no vectors).
        dim: usize,
        /// Optional sparse vectors, one entry per chunk. Absent in records
        /// written before this field existed (serde default keeps old WALs
        /// readable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sparse: Option<Vec<Option<ragvault_core::SparseVector>>>,
    },
    DeleteDocument {
        document_id: String,
    },
}

#[derive(Debug)]
pub struct WalRecord {
    pub seq: u64,
    pub op: WalOp,
    pub payload: Vec<f32>,
}

pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    policy: SyncPolicy,
}

impl Wal {
    pub fn wal_path(dir: &Path) -> PathBuf {
        dir.join("wal.log")
    }

    /// Open (or create) the WAL for appending.
    ///
    /// Opened read+write (not `append`) and seeked to the end: appends are
    /// serialized by the single-writer directory lock, so OS-level atomic
    /// append is unnecessary — and on Windows a handle opened in append mode
    /// lacks `FILE_WRITE_DATA`, which makes [`Self::truncate`]'s `set_len`
    /// fail with "Access is denied" (os error 5). Read+write keeps `set_len`
    /// portable across platforms.
    pub fn open(dir: &Path, policy: SyncPolicy) -> Result<Wal> {
        let path = Self::wal_path(dir);
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false) // keep existing records; we seek to the end below
            .open(&path)
            .map_err(|e| Error::io(format!("open wal {}", path.display()), e))?;
        // Position at end so records append after any existing log.
        file.seek(SeekFrom::End(0))
            .map_err(|e| Error::io("seek wal to end", e))?;
        Ok(Wal {
            path,
            writer: BufWriter::new(file),
            policy,
        })
    }

    /// Append one operation. Returns after the record is durable according
    /// to the sync policy.
    pub fn append(&mut self, seq: u64, op: &WalOp, payload: &[f32]) -> Result<()> {
        let header = serde_json::to_vec(op)?;
        let payload_bytes: Vec<u8> = payload.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&seq.to_le_bytes());
        hasher.update(&header);
        hasher.update(&payload_bytes);
        let crc = hasher.finalize();

        let w = &mut self.writer;
        w.write_all(&(header.len() as u32).to_le_bytes())
            .and_then(|_| w.write_all(&(payload_bytes.len() as u32).to_le_bytes()))
            .and_then(|_| w.write_all(&seq.to_le_bytes()))
            .and_then(|_| w.write_all(&header))
            .and_then(|_| w.write_all(&payload_bytes))
            .and_then(|_| w.write_all(&crc.to_le_bytes()))
            .map_err(|e| Error::io("append wal record", e))?;
        self.writer.flush().map_err(|e| Error::io("flush wal", e))?;
        if self.policy == SyncPolicy::Sync {
            self.sync()?;
        }
        Ok(())
    }

    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush().map_err(|e| Error::io("flush wal", e))?;
        self.writer
            .get_ref()
            .sync_data()
            .map_err(|e| Error::io("fsync wal", e))
    }

    /// Read all valid records with seq > `after_seq`; truncate a corrupt or
    /// torn tail. Returns records in order.
    pub fn replay(dir: &Path, after_seq: u64) -> Result<Vec<WalRecord>> {
        let path = Self::wal_path(dir);
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::io(format!("open wal {}", path.display()), e)),
        };
        let file_len = file.metadata().map_err(|e| Error::io("stat wal", e))?.len();
        let mut reader = BufReader::new(file);
        let mut records = Vec::new();
        let mut good_offset: u64 = 0;
        loop {
            match Self::read_record(&mut reader, file_len, good_offset) {
                Ok(Some((record, next_offset))) => {
                    if record.seq > after_seq {
                        records.push(record);
                    }
                    good_offset = next_offset;
                }
                Ok(None) => break,
                Err(_) => break, // corrupt tail — truncate below
            }
        }
        if good_offset < file_len {
            // Torn/corrupt tail: truncate to the last good record.
            let f = OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(|e| Error::io("open wal for truncate", e))?;
            f.set_len(good_offset)
                .map_err(|e| Error::io("truncate wal", e))?;
            f.sync_all().map_err(|e| Error::io("fsync wal", e))?;
        }
        Ok(records)
    }

    fn read_record(
        reader: &mut BufReader<File>,
        file_len: u64,
        offset: u64,
    ) -> Result<Option<(WalRecord, u64)>> {
        const FIXED: u64 = 4 + 4 + 8 + 4; // lens + seq + crc
        if offset + FIXED > file_len {
            return Ok(None);
        }
        let mut lens = [0u8; 16];
        if reader.read_exact(&mut lens).is_err() {
            return Ok(None);
        }
        let header_len = u32::from_le_bytes(lens[0..4].try_into().expect("4 bytes")) as u64;
        let payload_len = u32::from_le_bytes(lens[4..8].try_into().expect("4 bytes")) as u64;
        let seq = u64::from_le_bytes(lens[8..16].try_into().expect("8 bytes"));
        let total = offset + FIXED + header_len + payload_len;
        if header_len > 256 * 1024 * 1024
            || payload_len > 4 * 1024 * 1024 * 1024
            || total > file_len
        {
            return Ok(None); // implausible sizes = torn tail
        }
        let mut header = vec![0u8; header_len as usize];
        let mut payload_bytes = vec![0u8; payload_len as usize];
        let mut crc_bytes = [0u8; 4];
        if reader.read_exact(&mut header).is_err()
            || reader.read_exact(&mut payload_bytes).is_err()
            || reader.read_exact(&mut crc_bytes).is_err()
        {
            return Ok(None);
        }
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&seq.to_le_bytes());
        hasher.update(&header);
        hasher.update(&payload_bytes);
        if hasher.finalize() != u32::from_le_bytes(crc_bytes) {
            return Err(Error::corrupt(
                "wal",
                format!("crc mismatch at offset {offset}"),
            ));
        }
        let op: WalOp = serde_json::from_slice(&header)?;
        if !payload_bytes.len().is_multiple_of(4) {
            return Err(Error::corrupt("wal", "payload not a multiple of 4 bytes"));
        }
        let payload: Vec<f32> = payload_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();
        Ok(Some((WalRecord { seq, op, payload }, total)))
    }

    /// Reset the WAL after a successful snapshot publish: truncate to zero.
    pub fn truncate(&mut self) -> Result<()> {
        self.writer.flush().map_err(|e| Error::io("flush wal", e))?;
        let f = self.writer.get_mut();
        f.set_len(0).map_err(|e| Error::io("truncate wal", e))?;
        f.seek(SeekFrom::Start(0))
            .map_err(|e| Error::io("seek wal", e))?;
        f.sync_all().map_err(|e| Error::io("fsync wal", e))?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc(id: &str) -> Document {
        Document {
            document_id: id.into(),
            source_id: None,
            current_version: 1,
            title: None,
            metadata: json!({}),
        }
    }

    fn upsert(id: &str, dim: usize) -> WalOp {
        WalOp::UpsertDocument {
            document: doc(id),
            chunks: vec![],
            dim,
            sparse: None,
        }
    }

    #[test]
    fn records_without_sparse_field_still_parse() {
        // Backward compatibility: headers written before the `sparse` field
        // existed must keep replaying.
        let old_header = serde_json::json!({
            "op": "upsert_document",
            "document": doc("legacy"),
            "chunks": [],
            "dim": 0,
        });
        let parsed: WalOp = serde_json::from_value(old_header).unwrap();
        match parsed {
            WalOp::UpsertDocument { sparse, .. } => assert!(sparse.is_none()),
            _ => panic!("wrong op"),
        }
    }

    #[test]
    fn append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(1, &upsert("a", 2), &[1.0, 2.0]).unwrap();
        wal.append(
            2,
            &WalOp::DeleteDocument {
                document_id: "a".into(),
            },
            &[],
        )
        .unwrap();
        drop(wal);

        let records = Wal::replay(dir.path(), 0).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[0].payload, vec![1.0, 2.0]);
        assert!(matches!(records[1].op, WalOp::DeleteDocument { .. }));

        // after_seq filters already-snapshotted records
        let records = Wal::replay(dir.path(), 1).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 2);
    }

    #[test]
    fn torn_tail_is_truncated_and_replay_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(1, &upsert("a", 0), &[]).unwrap();
        wal.append(2, &upsert("b", 0), &[]).unwrap();
        drop(wal);

        // Simulate a crash mid-write: append garbage half-record.
        let path = Wal::wal_path(dir.path());
        let full_len = std::fs::metadata(&path).unwrap().len();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0x12, 0x34, 0x56]).unwrap();
        drop(f);

        let records = Wal::replay(dir.path(), 0).unwrap();
        assert_eq!(records.len(), 2, "good prefix survives");
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            full_len,
            "tail truncated"
        );

        // replay again: same result (idempotent)
        let records2 = Wal::replay(dir.path(), 0).unwrap();
        assert_eq!(records2.len(), 2);
    }

    #[test]
    fn corrupt_record_stops_replay_at_last_good() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(1, &upsert("a", 0), &[]).unwrap();
        let good_len = std::fs::metadata(Wal::wal_path(dir.path())).unwrap().len();
        wal.append(2, &upsert("b", 0), &[]).unwrap();
        drop(wal);

        // Flip a byte inside the second record's header.
        let path = Wal::wal_path(dir.path());
        let mut bytes = std::fs::read(&path).unwrap();
        let target = good_len as usize + 20;
        bytes[target] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let records = Wal::replay(dir.path(), 0).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 1);
    }

    #[test]
    fn truncate_resets_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(1, &upsert("a", 0), &[]).unwrap();
        wal.truncate().unwrap();
        wal.append(2, &upsert("b", 0), &[]).unwrap();
        drop(wal);
        let records = Wal::replay(dir.path(), 0).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 2);
    }

    #[test]
    fn reopen_appends_after_existing_records() {
        // Regression: the WAL is opened read+write (not O_APPEND) and seeked
        // to the end, so a reopened log must append after existing records
        // rather than overwrite from offset 0. (O_APPEND was dropped because
        // it makes set_len fail on Windows.)
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(1, &upsert("a", 0), &[1.0]).unwrap();
        drop(wal);

        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(2, &upsert("b", 0), &[2.0]).unwrap();
        drop(wal);

        let records = Wal::replay(dir.path(), 0).unwrap();
        assert_eq!(records.len(), 2, "reopen must not clobber the first record");
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[0].payload, vec![1.0]);
        assert_eq!(records[1].seq, 2);
        assert_eq!(records[1].payload, vec![2.0]);
    }

    #[test]
    fn truncate_then_reopen_then_append_is_clean() {
        // Exercises the full close-path shape that failed on Windows:
        // truncate (set_len 0) on the writer handle, reopen, append.
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(1, &upsert("a", 0), &[]).unwrap();
        wal.append(2, &upsert("b", 0), &[]).unwrap();
        wal.truncate().unwrap();
        drop(wal);

        let mut wal = Wal::open(dir.path(), SyncPolicy::Sync).unwrap();
        wal.append(3, &upsert("c", 0), &[]).unwrap();
        drop(wal);

        let records = Wal::replay(dir.path(), 0).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 3);
    }

    #[test]
    fn missing_wal_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Wal::replay(dir.path(), 0).unwrap().is_empty());
    }
}
