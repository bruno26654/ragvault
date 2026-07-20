# ADR 0016 — Storage v2: segmented binary format

Status: **accepted, not yet implemented** (scopes the v0.2 work deferred by
ADR 0004). Supersedes ADR 0004's "single mutable arena" decision when built.

## Contexto

v0.1 (ADR 0004) persists a snapshot generation as `gen-N/state.json` (the full
document/chunk/version/index state as one JSON blob) plus `gen-N/vectors.bin`
(raw f32 LE, already binary and mmap-able), published atomically via a
checksummed `manifest.json` and rename. This is durable and crash-safe
(prepared-write + publish-at-end, ADR 0006), but:

- **Every publish rewrites the entire metadata state as JSON.** Cost grows with
  total corpus size, not with the delta — O(N) snapshots.
- **Compaction rewrites arena + indexes synchronously**, pausing writers.
- **No immutable-segment reuse**: unchanged data is re-serialized each snapshot.

The current foundation to build on (already present, keep it): versioned
manifest, atomic rename publish, per-file CRC32, generation GC only after the
new manifest is durable, and the `RwLock`-isolated logical snapshot the engine
already exposes to queries.

## Decisão (v2 target)

Replace the monolithic `state.json` with a **segmented binary state**:

1. **Segments.** State is a set of immutable segments plus one mutable segment.
   - Immutable segment = binary file `seg-<id>.rvseg`: a length-prefixed,
     CRC-per-record log of `document`/`chunk`/`version`/`tombstone` records,
     with a trailing footer holding a streaming CRC over the whole file and the
     record count. Written once, never mutated.
   - Mutable segment = the in-memory tail (backed by the WAL) that becomes a
     new immutable segment at flush.
2. **Manifest v2.** `manifest.json` (format_version = 2) lists the ordered set
   of live segments (`{file, crc32, records, min_seq, max_seq}`) plus the
   vector file(s). Publish stays atomic-rename; adding a segment writes the new
   `seg-*.rvseg`, fsyncs it, then writes+renames a manifest that references it.
   A crash before the manifest rename leaves an orphan segment that GC removes.
3. **Multi-segment query.** Reads merge live records across segments newest-
   first; a tombstone or a newer version in a later segment shadows older ones.
   The engine's logical-snapshot handle pins the segment set for a query's
   lifetime, so a concurrent compaction never pulls a file out from under a
   reader.
4. **Streaming checksums.** CRC is computed incrementally while writing (no
   second pass over the buffer) and verified incrementally on load; a bad
   segment fails the open with `Corrupt` naming the segment and offset.
5. **Safe compaction.** Compaction merges several immutable segments into one
   new segment off to the side, fsyncs it, then publishes a manifest that
   swaps the inputs for the output. Old segments are deleted only after the new
   manifest is durable and no pinned reader still references them. This makes
   compaction crash-safe and read-safe without a global write pause.
6. **Format compatibility + migration.** A v1 vault (single `state.json`,
   format_version = 1) opens read-only-compatible: on first flush it is
   rewritten as a single v2 immutable segment. `manifest.format_version > 2`
   still fails closed (as today). Downgrade is not supported and is documented.

## Alternativas

- Full LSM (leveled compaction, bloom filters) — more than a single-node
  embedded store needs; rejected for v1.0 (ADR 0004's reasoning still holds).
- Keep JSON but diff it — fragile and still O(N) to load; rejected.

## Consequências

- Snapshots become O(delta); large corpora stop paying full re-serialization.
- Compaction no longer requires a global write pause.
- More moving parts in the durable layer — mitigated by the test plan below,
  which is a hard gate (this is the subsystem that had the P0 in ADR 0006).

## Critérios de aceitação (test plan — all required before marking validated)

1. Round-trip: write → reopen → identical documents/chunks/versions/search
   results, across ≥2 segments (differential harness extended).
2. Tombstone/version shadowing correct across segment boundaries.
3. Crash injection: kill between (a) segment fsync and manifest rename,
   (b) manifest rename and old-segment GC — reopen is consistent both times,
   orphans are GC'd, no partial state visible.
4. Read-during-compaction: a reader holding a pinned snapshot returns stable
   results while a compaction publishes and GCs; the reader's segments are not
   deleted until it releases.
5. Streaming-CRC corruption of any segment byte fails open with an actionable
   `Corrupt` error naming file + offset.
6. v1→v2 migration: an existing v1 vault opens, and first flush produces a
   valid v2 manifest with one segment; results unchanged before/after.
7. `cargo test` + `differential_consistency.rs` extended to include a
   multi-segment config; fmt/clippy clean.

## Validação

Not yet implemented. Until the test plan above is green, the engine keeps the
v0.1 single-arena format (ADR 0004). `docs/STORAGE.md` and `TASKS.md` track
this as the one remaining P2 architectural item for v1.0.
