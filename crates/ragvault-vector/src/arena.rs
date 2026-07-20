//! Contiguous row-major vector arena.
//!
//! Vectors live in one contiguous `Vec<f32>` (hot data) separate from chunk
//! metadata (cold data, materialized only after selection). Internal ids are
//! dense `u32` row indices; deletions are tombstoned in a bitmap and physical
//! space is reclaimed on compaction.

use std::sync::Arc;

use ragvault_core::{Error, Metric, Result};

use crate::kernels;

/// Read-only mmap-backed prefix of the arena (rows 0..rows). The f32 view is
/// obtained with `bytemuck::try_cast_slice` — safe, alignment- and
/// length-checked (mmaps are page-aligned, so the cast never fails after the
/// length validation in `from_mmap`).
#[derive(Debug, Clone)]
struct MmapBase {
    mmap: Arc<memmap2::Mmap>,
    rows: usize,
}

impl MmapBase {
    #[inline]
    fn as_f32(&self) -> &[f32] {
        bytemuck::try_cast_slice(&self.mmap[..]).expect("validated in from_mmap")
    }
}

#[derive(Debug, Clone)]
pub struct VectorArena {
    dim: usize,
    metric: Metric,
    /// Optional mmap-backed prefix (storage="mmap"); owned rows follow it.
    base: Option<MmapBase>,
    data: Vec<f32>,
    deleted: Vec<bool>,
    live: usize,
}

impl VectorArena {
    pub fn new(dim: usize, metric: Metric) -> Self {
        VectorArena {
            dim,
            metric,
            base: None,
            data: Vec::new(),
            deleted: Vec::new(),
            live: 0,
        }
    }

    fn base_rows(&self) -> usize {
        self.base.as_ref().map(|b| b.rows).unwrap_or(0)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Total rows including tombstones.
    pub fn len(&self) -> usize {
        self.deleted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.deleted.is_empty()
    }

    /// Live (non-deleted) rows.
    pub fn live(&self) -> usize {
        self.live
    }

    pub fn is_deleted(&self, id: u32) -> bool {
        self.deleted.get(id as usize).copied().unwrap_or(true)
    }

    /// Append a vector, returning its internal id. Cosine-metric vectors are
    /// normalized on insert so search reduces to dot products.
    pub fn push(&mut self, vector: &[f32]) -> Result<u32> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid(
                "vector",
                "finite f32 values",
                "NaN or infinity",
            ));
        }
        let id = self.deleted.len() as u32;
        let start = self.data.len();
        self.data.extend_from_slice(vector);
        if self.metric == Metric::Cosine {
            kernels::normalize(&mut self.data[start..start + self.dim]);
        }
        self.deleted.push(false);
        self.live += 1;
        Ok(id)
    }

    pub fn delete(&mut self, id: u32) -> bool {
        if let Some(slot) = self.deleted.get_mut(id as usize) {
            if !*slot {
                *slot = true;
                self.live -= 1;
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn get(&self, id: u32) -> &[f32] {
        let idx = id as usize;
        let base_rows = self.base_rows();
        if idx < base_rows {
            let base = self.base.as_ref().expect("base_rows > 0 implies base");
            let start = idx * self.dim;
            &base.as_f32()[start..start + self.dim]
        } else {
            let start = (idx - base_rows) * self.dim;
            &self.data[start..start + self.dim]
        }
    }

    /// Similarity score (higher = better) between a stored row and a query.
    /// Cosine rows are pre-normalized, so cosine score expects a normalized
    /// query and uses a plain dot product.
    #[inline]
    pub fn score(&self, id: u32, query: &[f32]) -> f32 {
        let row = self.get(id);
        match self.metric {
            Metric::Cosine | Metric::Dot => kernels::dot(row, query),
            Metric::L2 => -kernels::l2_sq(row, query),
        }
    }

    /// Prepare a query vector for this arena (normalizes under cosine).
    pub fn prepare_query(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut q = query.to_vec();
        if q.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid(
                "query vector",
                "finite f32 values",
                "NaN or infinity",
            ));
        }
        if self.metric == Metric::Cosine {
            kernels::normalize(&mut q);
        }
        Ok(q)
    }

    /// Storage parts for persistence: (mmap-backed prefix, owned tail).
    /// Concatenated they are the full row-major arena.
    pub fn vector_parts(&self) -> (&[f32], &[f32]) {
        match &self.base {
            Some(base) => (base.as_f32(), &self.data),
            None => (&[], &self.data),
        }
    }

    pub fn deleted_bitmap(&self) -> &[bool] {
        &self.deleted
    }

    /// Rebuild an arena from persisted parts.
    pub fn from_parts(
        dim: usize,
        metric: Metric,
        data: Vec<f32>,
        deleted: Vec<bool>,
    ) -> Result<Self> {
        if dim == 0 || data.len() != dim * deleted.len() {
            return Err(Error::corrupt(
                "vector arena",
                format!(
                    "data length {} inconsistent with dim {} x rows {}",
                    data.len(),
                    dim,
                    deleted.len()
                ),
            ));
        }
        let live = deleted.iter().filter(|d| !**d).count();
        Ok(VectorArena {
            dim,
            metric,
            base: None,
            data,
            deleted,
            live,
        })
    }

    /// Build an arena whose stored rows are served from a read-only mmap
    /// (storage="mmap"). New rows appended afterwards live in RAM.
    pub fn from_mmap(
        dim: usize,
        metric: Metric,
        mmap: memmap2::Mmap,
        deleted: Vec<bool>,
    ) -> Result<Self> {
        let bytes = mmap.len();
        if dim == 0 || bytes != dim * deleted.len() * 4 {
            return Err(Error::corrupt(
                "vector arena (mmap)",
                format!(
                    "file has {bytes} bytes, expected dim {dim} x rows {} x 4",
                    deleted.len()
                ),
            ));
        }
        if bytemuck::try_cast_slice::<u8, f32>(&mmap[..]).is_err() {
            return Err(Error::corrupt(
                "vector arena (mmap)",
                "mapping is not 4-byte aligned",
            ));
        }
        let live = deleted.iter().filter(|d| !**d).count();
        let rows = deleted.len();
        Ok(VectorArena {
            dim,
            metric,
            base: Some(MmapBase {
                mmap: Arc::new(mmap),
                rows,
            }),
            data: Vec::new(),
            deleted,
            live,
        })
    }

    /// True when the stored prefix is served from an mmap.
    pub fn is_mmap(&self) -> bool {
        self.base.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_get_delete() {
        let mut arena = VectorArena::new(3, Metric::Dot);
        let a = arena.push(&[1.0, 0.0, 0.0]).unwrap();
        let b = arena.push(&[0.0, 2.0, 0.0]).unwrap();
        assert_eq!(arena.live(), 2);
        assert_eq!(arena.get(b), &[0.0, 2.0, 0.0]);
        assert!(arena.delete(a));
        assert!(!arena.delete(a), "double delete is a no-op");
        assert_eq!(arena.live(), 1);
        assert!(arena.is_deleted(a));
    }

    #[test]
    fn cosine_normalizes_on_insert() {
        let mut arena = VectorArena::new(2, Metric::Cosine);
        let id = arena.push(&[3.0, 4.0]).unwrap();
        let row = arena.get(id);
        assert!((row[0] - 0.6).abs() < 1e-6);
        assert!((row[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_vectors() {
        let mut arena = VectorArena::new(3, Metric::Cosine);
        assert!(arena.push(&[1.0, 2.0]).is_err());
        assert!(arena.push(&[1.0, f32::NAN, 0.0]).is_err());
        assert!(arena.prepare_query(&[1.0]).is_err());
    }

    #[test]
    fn l2_scores_are_negated_distances() {
        let mut arena = VectorArena::new(2, Metric::L2);
        let id = arena.push(&[0.0, 0.0]).unwrap();
        assert_eq!(arena.score(id, &[3.0, 4.0]), -25.0);
    }
}
