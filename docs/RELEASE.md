# Release readiness

This document is the gate between the current `main` and a public release of
RagVault. It records what "ready" means, what is verified in CI, and what is
still explicitly out of scope. Nothing here authorizes publishing: **no PyPI
upload and no public GitHub release happens without an explicit human
decision.**

> **Status — 1.0.0-rc1 (main `22e880d`, 2026-07-27):** every Section 2 gate is
> green: fmt/clippy clean; 122 Rust + 89 Python tests; real integration
> roundtrips at pinned versions; wheels built for all four platforms with
> clean-venv install smoke (plus a local re-run of wheel/CLI/compaction
> smokes on this tree). Versions: Rust `1.0.0-rc1` / Python `1.0.0rc1`.
> Externally pending (not a code gate): real-GPU validation (docs/GPU.md).
> The semantic eval rows have since been executed on a machine with model
> access and are recorded with measured numbers in benchmarks/RESULTS-RAG.md.
> The `v1.0.0-rc1` tag and any publication remain manual steps per §6.

## 1. Supported scope for v1.0

- **Target:** single-node, CPU. Local-first, embedded (no server).
- **Python:** 3.9–3.12, `abi3` wheels (one wheel per platform covers all
  supported minors).
- **Platforms with wheels + smoke test in CI:** Linux x86-64, Linux aarch64,
  macOS Apple Silicon (arm64), Windows x86-64.
- **Experimental / not gating:** cuVS/CAGRA GPU sidecar (no GPU hardware in
  CI; runbook in [GPU.md](GPU.md)), and any non-CPU acceleration.

## 2. Release gates

A release candidate must have all of the following green. Each maps to a CI
job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

| Gate | Evidence | Job |
|---|---|---|
| `cargo fmt --check` clean | no diff | `rust` |
| `cargo clippy -D warnings` clean | no warnings | `rust` |
| Rust tests (unit + differential + proptest) pass | test output | `rust` |
| Python tests pass on 3.9 / 3.11 / 3.12 | pytest | `python` |
| CLI smoke (`init`/`sync`/`query`/`doctor`) | CLI job | `python` |
| Real integration roundtrips (LangChain, LlamaIndex, Haystack, DSPy) at pinned versions | pytest | `integrations` |
| Wheels build for all four platforms | artifacts | `wheels` |
| Clean-venv install smoke (`open → add → retrieve`) per platform | smoke step | `wheels` |
| CHANGELOG updated | [CHANGELOG.md](../CHANGELOG.md) | manual |

The pinned integration versions live in the `integrations` job so an upstream
breaking change fails CI instead of silently rotting. Bump them deliberately.

## 3. Versioning and compatibility

- **SemVer.** While `0.x`, the documented Python API
  ([PYTHON-API.md](PYTHON-API.md)) is treated as stable; breaking changes to it
  are called out in the CHANGELOG with a migration note. Internal Rust crate
  APIs and unstable on-disk details may still change.
- **On-disk format.** The vault format is versioned. A newer library reads an
  older vault; format changes ship with a migration path (see
  [STORAGE.md](STORAGE.md)). Never silently rewrite a vault the running version
  cannot fully understand — fail with an actionable error instead.

## 4. Backup and restore

A vault is a self-contained directory. To back up: stop writers (the
single-writer lock guarantees no concurrent writer), then copy the directory.
To restore: copy it back and `ragvault.open(path)`. Because publishes are
atomic (snapshot generations + CRC32 WAL with torn-tail truncation), a copy
taken while a writer was mid-publish reopens at the last durable state — a
partially written batch is never observable. `ragvault doctor <path>` verifies
integrity after a restore.

## 5. Security and limitations

- **Untrusted documents.** Parsers run on caller-provided bytes; treat
  document ingestion as parsing untrusted input. No parser executes document
  content. Optional parsers (PDF/DOCX/HTML) pull third-party libraries — pin
  and audit those in your own environment.
- **No network at rest.** RagVault performs no outbound network calls on its
  own; embedding downloads only happen if you configure a downloading
  embedder. `offline-lite` needs no network.
- **Known limitations for v1.0:** single-node only (no distributed/remote —
  `ragvault.connect()` is a reserved signature that raises `NotImplementedError`);
  GPU path experimental and unvalidated on hardware; compaction is synchronous
  (background compaction deferred); Roaring bitmaps / OPQ / binary quantization
  are deferred behind their own benchmark gate.

## 6. Release procedure (requires explicit authorization)

1. Confirm all Section 2 gates are green on `main`.
2. Update `[Unreleased]` in the CHANGELOG to the target version + date.
3. Bump `version` in the workspace `Cargo.toml` and `pyproject.toml`.
4. Tag `vX.Y.Z`; CI builds the wheel artifacts.
5. **Only after human sign-off:** publish wheels to PyPI and cut the GitHub
   release from the tag. Do not automate this step.
