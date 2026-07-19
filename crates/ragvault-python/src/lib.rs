//! PyO3 bindings: exposes the vault engine to Python as `ragvault._native`.
//!
//! - All engine calls that do real work run inside `py.allow_threads`, so
//!   the GIL is released during search, ingestion, flush and compaction
//!   (validated by a threading test in the Python suite).
//! - Rust panics never cross the FFI boundary (PyO3 converts them to
//!   `PanicException`); expected failures are converted into specific,
//!   informative Python exceptions defined in `ragvault.errors`.

use std::path::PathBuf;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use ragvault_core::{Chunk, Document, Error, SparseVector};
use ragvault_engine::{EngineConfig, SearchRequest, VaultEngine};

create_exception!(
    ragvault_native,
    VaultError,
    PyRuntimeError,
    "Base error for RagVault storage/engine failures."
);
create_exception!(
    ragvault_native,
    VaultLockedError,
    VaultError,
    "The vault directory is locked by another writer."
);
create_exception!(
    ragvault_native,
    VaultCorruptError,
    VaultError,
    "Persisted data failed checksum or format validation."
);
create_exception!(
    ragvault_native,
    DimensionMismatchError,
    PyValueError,
    "Vector dimension differs from the vault's configured dimension."
);

fn to_py_err(err: Error) -> PyErr {
    match &err {
        Error::Locked { .. } => VaultLockedError::new_err(err.to_string()),
        Error::Corrupt { .. } | Error::IncompatibleFormat { .. } => {
            VaultCorruptError::new_err(err.to_string())
        }
        Error::DimensionMismatch { .. } => DimensionMismatchError::new_err(err.to_string()),
        Error::InvalidInput { .. } | Error::InvalidFilter(_) => {
            PyValueError::new_err(err.to_string())
        }
        _ => VaultError::new_err(err.to_string()),
    }
}

fn json_loads(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    let json_mod = PyModule::import(py, "json")?;
    let obj = json_mod.call_method1(
        "loads",
        (serde_json::to_string(value)
            .map_err(|e| PyRuntimeError::new_err(format!("serialize internal value: {e}")))?,),
    )?;
    Ok(obj.unbind())
}

fn parse_json(text: &str, what: &str) -> PyResult<serde_json::Value> {
    serde_json::from_str(text)
        .map_err(|e| PyValueError::new_err(format!("invalid JSON for {what}: {e}")))
}

/// The native vault handle. One writer per directory; safe to share across
/// Python threads.
#[pyclass(name = "Vault")]
struct PyVault {
    engine: Option<VaultEngine>,
    path: PathBuf,
}

impl PyVault {
    fn engine(&self) -> PyResult<&VaultEngine> {
        self.engine
            .as_ref()
            .ok_or_else(|| VaultError::new_err("vault is closed"))
    }
}

#[pymethods]
impl PyVault {
    /// Open or create a vault. `config_json` carries EngineConfig fields.
    #[staticmethod]
    fn open(py: Python<'_>, path: String, config_json: String) -> PyResult<Self> {
        let config: EngineConfig = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid engine config: {e}")))?;
        let path = PathBuf::from(path);
        let engine = py
            .allow_threads(|| VaultEngine::open(&path, config))
            .map_err(to_py_err)?;
        Ok(PyVault {
            engine: Some(engine),
            path,
        })
    }

    /// Upsert a document with chunks and dense vectors.
    /// `document_json` — Document; `chunks_json` — list[Chunk];
    /// `vectors` — float32 [n_chunks, dim]; `sparse_json` — optional
    /// list of {"indices": [...], "values": [...]} or null entries.
    #[pyo3(signature = (document_json, chunks_json, vectors, sparse_json=None))]
    fn upsert_document(
        &self,
        py: Python<'_>,
        document_json: String,
        chunks_json: String,
        vectors: PyReadonlyArray2<'_, f32>,
        sparse_json: Option<String>,
    ) -> PyResult<u64> {
        let document: Document = serde_json::from_str(&document_json)
            .map_err(|e| PyValueError::new_err(format!("invalid document: {e}")))?;
        let chunks: Vec<Chunk> = serde_json::from_str(&chunks_json)
            .map_err(|e| PyValueError::new_err(format!("invalid chunks: {e}")))?;
        let sparse: Option<Vec<Option<SparseVector>>> = match sparse_json {
            Some(s) => serde_json::from_str(&s)
                .map_err(|e| PyValueError::new_err(format!("invalid sparse vectors: {e}")))?,
            None => None,
        };
        let array = vectors.as_array();
        if array.nrows() != chunks.len() {
            return Err(PyValueError::new_err(format!(
                "vectors has {} rows but {} chunks were provided",
                array.nrows(),
                chunks.len()
            )));
        }
        let flat: Vec<f32> = array.iter().copied().collect();
        let engine = self.engine()?;
        py.allow_threads(|| engine.upsert_document(document, chunks, &flat, sparse))
            .map_err(to_py_err)
    }

    fn delete_document(&self, py: Python<'_>, document_id: String) -> PyResult<bool> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.delete_document(&document_id))
            .map_err(to_py_err)
    }

    /// Run a search. `request_json` is a SearchRequest without the dense
    /// vector; the vector rides separately as a numpy float32 array. Note:
    /// the vector IS copied once at the boundary (into a `Vec<f32>`) — this
    /// is not zero-copy, and we do not claim it is.
    #[pyo3(signature = (request_json, vector=None))]
    fn search(
        &self,
        py: Python<'_>,
        request_json: String,
        vector: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<PyObject> {
        let mut request: SearchRequest = serde_json::from_str(&request_json)
            .map_err(|e| PyValueError::new_err(format!("invalid search request: {e}")))?;
        if let Some(v) = vector {
            request.vector = Some(v.as_array().iter().copied().collect());
        }
        let engine = self.engine()?;
        let response = py
            .allow_threads(|| engine.search(&request))
            .map_err(to_py_err)?;
        let value = serde_json::to_value(&response)
            .map_err(|e| PyRuntimeError::new_err(format!("serialize response: {e}")))?;
        json_loads(py, &value)
    }

    fn get_chunk(&self, py: Python<'_>, chunk_id: String) -> PyResult<Option<PyObject>> {
        match self.engine()?.get_chunk(&chunk_id) {
            Some(chunk) => {
                let value = serde_json::to_value(&chunk)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(Some(json_loads(py, &value)?))
            }
            None => Ok(None),
        }
    }

    fn get_document(&self, py: Python<'_>, document_id: String) -> PyResult<Option<PyObject>> {
        match self.engine()?.get_document(&document_id) {
            Some(doc) => {
                let value = serde_json::to_value(&doc)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(Some(json_loads(py, &value)?))
            }
            None => Ok(None),
        }
    }

    fn get_document_chunks(&self, py: Python<'_>, document_id: String) -> PyResult<PyObject> {
        let chunks = self.engine()?.get_document_chunks(&document_id);
        let value =
            serde_json::to_value(&chunks).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        json_loads(py, &value)
    }

    fn list_documents(&self, py: Python<'_>) -> PyResult<PyObject> {
        let docs = self.engine()?.list_documents();
        let value =
            serde_json::to_value(&docs).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        json_loads(py, &value)
    }

    fn list_document_versions(&self, py: Python<'_>, document_id: String) -> PyResult<PyObject> {
        let versions = self.engine()?.list_document_versions(&document_id);
        let value =
            serde_json::to_value(&versions).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        json_loads(py, &value)
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.flush()).map_err(to_py_err)
    }

    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.compact()).map_err(to_py_err)
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.engine()?.stats();
        json_loads(py, &stats)
    }

    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.engine()?.config())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Flush and release the writer lock.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(engine) = self.engine.take() {
            py.allow_threads(|| engine.close()).map_err(to_py_err)?;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Vault(path={:?}, open={})",
            self.path,
            self.engine.is_some()
        )
    }
}

/// Validate a filter expression without running a query (used by the Python
/// layer for early, clear errors).
#[pyfunction]
fn validate_filter(filter_json: String) -> PyResult<()> {
    let value = parse_json(&filter_json, "filter")?;
    ragvault_core::Filter::parse(&value)
        .map(|_| ())
        .map_err(to_py_err)
}

#[pymodule]
#[pyo3(name = "_native")]
fn ragvault_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVault>()?;
    m.add_function(wrap_pyfunction!(validate_filter, m)?)?;
    m.add("VaultError", py.get_type::<VaultError>())?;
    m.add("VaultLockedError", py.get_type::<VaultLockedError>())?;
    m.add("VaultCorruptError", py.get_type::<VaultCorruptError>())?;
    m.add(
        "DimensionMismatchError",
        py.get_type::<DimensionMismatchError>(),
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
