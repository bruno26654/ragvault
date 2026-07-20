//! Binary segment container (storage v2, ADR 0016).
//!
//! An append-only, immutable-once-finalized log of length-prefixed records.
//! Every record carries its own CRC32; the file ends with a footer holding the
//! record count and a streaming CRC computed over the whole body. Both are
//! verified incrementally on read, so a torn write, a bit flip, or a truncated
//! file fails the open with an actionable [`Error::corrupt`] naming the offset.
//!
//! Layout:
//!
//! ```text
//! [MAGIC "RVSG"][format u32 LE]
//! repeated:
//!   [payload_len u32 LE][payload bytes][record_crc32 u32 LE]
//! [FOOTER "RVSF"][record_count u64 LE][stream_crc32 u32 LE]
//! ```
//!
//! `stream_crc32` is the CRC over every byte from the start of the file up to
//! (but not including) the footer's own `stream_crc32` field. `record_crc32`
//! is the CRC over that record's payload bytes only.
//!
//! Record payloads are opaque `[u8]` at this layer: callers decide their
//! meaning (a base state blob, or a delta of upsert/delete operations). This
//! keeps the container reusable for both the immutable base segment and the
//! mutable-tail deltas described in ADR 0016.

use std::io::{BufWriter, Write};
use std::path::Path;

use ragvault_core::{Error, Result};

const MAGIC: &[u8; 4] = b"RVSG";
const FOOTER_MAGIC: &[u8; 4] = b"RVSF";

/// Current segment container format. Distinct from the manifest's
/// `format_version`; this versions the on-disk framing of a single segment.
pub const SEGMENT_FORMAT: u32 = 1;

/// Streaming writer for a segment file. Records are appended, then [`finish`]
/// writes the footer and fsyncs. Dropping without `finish` leaves an
/// unfinalized (footerless) file, which the reader rejects as corrupt.
pub struct SegmentWriter {
    inner: BufWriter<std::fs::File>,
    hasher: crc32fast::Hasher,
    records: u64,
}

impl SegmentWriter {
    /// Create (truncating) the segment file and write its header.
    pub fn create(path: &Path) -> Result<SegmentWriter> {
        let file = std::fs::File::create(path)
            .map_err(|e| Error::io(format!("create {}", path.display()), e))?;
        let mut w = SegmentWriter {
            inner: BufWriter::new(file),
            hasher: crc32fast::Hasher::new(),
            records: 0,
        };
        w.write_all(MAGIC)?;
        w.write_all(&SEGMENT_FORMAT.to_le_bytes())?;
        Ok(w)
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.hasher.update(bytes);
        self.inner
            .write_all(bytes)
            .map_err(|e| Error::io("write segment", e))
    }

    /// Append one record.
    pub fn append(&mut self, payload: &[u8]) -> Result<()> {
        let len = u32::try_from(payload.len())
            .map_err(|_| Error::invalid("segment record", "payload <= 4 GiB", "larger payload"))?;
        self.write_all(&len.to_le_bytes())?;
        self.write_all(payload)?;
        let record_crc = crc32_of(payload);
        self.write_all(&record_crc.to_le_bytes())?;
        self.records += 1;
        Ok(())
    }

    /// Write the footer, flush, and fsync. Consumes the writer.
    pub fn finish(mut self) -> Result<()> {
        self.write_all(FOOTER_MAGIC)?;
        self.write_all(&self.records.to_le_bytes())?;
        // stream_crc covers everything written so far (header + records +
        // footer magic + count), not the crc field itself.
        let stream_crc = self.hasher.clone().finalize();
        self.inner
            .write_all(&stream_crc.to_le_bytes())
            .map_err(|e| Error::io("write segment footer crc", e))?;
        self.inner
            .flush()
            .map_err(|e| Error::io("flush segment", e))?;
        let file = self
            .inner
            .into_inner()
            .map_err(|e| Error::io("finalize segment", e.into_error()))?;
        file.sync_all().map_err(|e| Error::io("fsync segment", e))?;
        Ok(())
    }
}

/// Read and fully verify a segment file, returning its record payloads.
///
/// Verification is incremental: each record's CRC is checked as it is read,
/// and the footer's streaming CRC and record count are checked at the end. Any
/// mismatch, a missing/short footer, or a bad magic is a [`Error::corrupt`].
pub fn read_records(path: &Path) -> Result<Vec<Vec<u8>>> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::io(format!("read {}", path.display()), e))?;
    let name = path.display().to_string();
    decode(&bytes, &name)
}

/// Verify and decode a segment already held in memory (same checks as
/// [`read_records`]). Useful when the caller has already read the file to
/// check a manifest-level checksum and wants to avoid a second read.
pub fn decode(bytes: &[u8], name: &str) -> Result<Vec<Vec<u8>>> {
    const FOOTER_LEN: usize = 4 + 8 + 4; // magic + count + stream_crc
    if bytes.len() < 8 + FOOTER_LEN {
        return Err(Error::corrupt(name, "segment shorter than header + footer"));
    }
    if &bytes[0..4] != MAGIC {
        return Err(Error::corrupt(name, "bad segment magic"));
    }
    let format = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
    if format != SEGMENT_FORMAT {
        return Err(Error::corrupt(
            name,
            format!("unsupported segment format {format}"),
        ));
    }

    let footer_start = bytes.len() - FOOTER_LEN;
    let footer = &bytes[footer_start..];
    if &footer[0..4] != FOOTER_MAGIC {
        return Err(Error::corrupt(name, "missing segment footer (torn write?)"));
    }
    let declared_count = u64::from_le_bytes(footer[4..12].try_into().expect("8 bytes"));
    let declared_crc = u32::from_le_bytes(footer[12..16].try_into().expect("4 bytes"));
    // stream_crc covers everything up to (not including) the crc field itself.
    let actual_crc = crc32_of(&bytes[..bytes.len() - 4]);
    if actual_crc != declared_crc {
        return Err(Error::corrupt(name, "segment stream crc mismatch"));
    }

    let mut records = Vec::new();
    let mut offset = 8usize;
    while offset < footer_start {
        if offset + 4 > footer_start {
            return Err(Error::corrupt(
                name,
                format!("truncated record length at offset {offset}"),
            ));
        }
        let len =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("4 bytes")) as usize;
        offset += 4;
        let end = offset + len;
        if end + 4 > footer_start {
            return Err(Error::corrupt(
                name,
                format!("record at offset {offset} overruns footer"),
            ));
        }
        let payload = &bytes[offset..end];
        let record_crc = u32::from_le_bytes(bytes[end..end + 4].try_into().expect("4 bytes"));
        if crc32_of(payload) != record_crc {
            return Err(Error::corrupt(
                name,
                format!("record crc mismatch at offset {offset}"),
            ));
        }
        records.push(payload.to_vec());
        offset = end + 4;
    }
    if records.len() as u64 != declared_count {
        return Err(Error::corrupt(
            name,
            format!(
                "segment record count mismatch: footer says {declared_count}, found {}",
                records.len()
            ),
        ));
    }
    Ok(records)
}

fn crc32_of(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_segment(path: &Path, payloads: &[&[u8]]) {
        let mut w = SegmentWriter::create(path).unwrap();
        for p in payloads {
            w.append(p).unwrap();
        }
        w.finish().unwrap();
    }

    #[test]
    fn round_trip_multiple_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvseg");
        write_segment(
            &path,
            &[b"alpha", b"", b"a longer payload here", &[0u8; 300]],
        );
        let got = read_records(&path).unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], b"alpha");
        assert_eq!(got[1], b"");
        assert_eq!(got[2], b"a longer payload here");
        assert_eq!(got[3], vec![0u8; 300]);
    }

    #[test]
    fn empty_segment_is_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvseg");
        write_segment(&path, &[]);
        assert_eq!(read_records(&path).unwrap().len(), 0);
    }

    #[test]
    fn record_bit_flip_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvseg");
        write_segment(&path, &[b"hello world", b"second record"]);
        let mut bytes = std::fs::read(&path).unwrap();
        // Flip a byte inside the first record's payload (after 8-byte header
        // + 4-byte length prefix).
        bytes[8 + 4 + 2] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let err = read_records(&path).unwrap_err();
        assert!(format!("{err}").contains("crc"), "got: {err}");
    }

    #[test]
    fn footer_corruption_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvseg");
        write_segment(&path, &[b"payload"]);
        let mut bytes = std::fs::read(&path).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF; // corrupt stream crc
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_records(&path).is_err());
    }

    #[test]
    fn truncated_footer_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvseg");
        write_segment(&path, &[b"payload"]);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 6); // eat part of the footer
        std::fs::write(&path, &bytes).unwrap();
        let err = read_records(&path).unwrap_err();
        assert!(
            format!("{err}").to_lowercase().contains("footer")
                || format!("{err}").to_lowercase().contains("crc")
        );
    }

    #[test]
    fn count_mismatch_is_detected() {
        // Hand-build a segment whose footer count lies.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.rvseg");
        let mut body = Vec::new();
        body.extend_from_slice(MAGIC);
        body.extend_from_slice(&SEGMENT_FORMAT.to_le_bytes());
        let payload = b"one";
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(payload);
        body.extend_from_slice(&crc32_of(payload).to_le_bytes());
        body.extend_from_slice(FOOTER_MAGIC);
        body.extend_from_slice(&2u64.to_le_bytes()); // lie: claim 2 records
        let crc = crc32_of(&body);
        body.extend_from_slice(&crc.to_le_bytes());
        std::fs::write(&path, &body).unwrap();
        let err = read_records(&path).unwrap_err();
        assert!(format!("{err}").contains("count mismatch"), "got: {err}");
    }
}
