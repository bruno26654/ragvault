# ADR 0016 — Storage v2: segmented binary format

Status: **implemented and validated** (the v0.2 work deferred by ADR 0004).
Supersedes ADR 0004's "single mutable arena" decision.

**Implemented:** the binary segment container (`crate::segment`), the v2 base
format (`format_version = 2` writes `gen-N/state.rvseg`) with transparent v1→v2
migration, and **multi-segment delta flush** — after the first full base, each
flush appends an O(delta) binary delta segment (WAL-shaped upsert/delete ops)
instead of rewriting the base; a budget of `MAX_DELTA_SEGMENTS` deltas triggers
a full base rewrite, and `compact()` collapses deltas into a fresh base. Open
applies base → deltas → live WAL in seq order.

**Read-during-compaction (acceptance #4):** implemented. The expensive
rebuild runs on a snapshot cloned under a short read lock, so concurrent
readers keep searching for the whole rebuild; the write lock is held only for
the final swap + durable publish. A concurrent write invalidates the snapshot
(detected by seq) — the rebuild retries off-lock, then falls back to building
under the write lock (always correct). Readers never reference on-disk files
directly (state lives in memory; an mmap keeps its old generation readable
even after GC), so segment GC can never pull data out from under a reader.

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

## Critérios de aceitação (test plan)

Status: ✅ done · ◻️ remaining.

1. ✅ Round-trip across ≥2 delta segments
   (`differential_multi_segment_deltas`, `delta_flush_accumulates_segments_and_reopens`).
2. ✅ Tombstone/version shadowing across segment boundaries
   (`delta_tombstone_and_replace_shadow_base`, and the multi-segment
   differential workload includes replaces + deletes spanning segments).
3. ✅ Crash injection: an orphan delta segment written but not referenced by
   the manifest (crash between segment fsync and manifest rename) is ignored on
   reopen (`orphan_delta_segment_is_ignored_on_reopen`); the manifest rename +
   fsync + GC-after-durable protocol from ADR 0004 is unchanged, and full-base
   GC removes superseded delta files only after the new manifest is durable.
4. ✅ Read-during-compaction: the rebuild runs off-lock on a cloned snapshot;
   readers search concurrently throughout and observe either fully-pre or
   fully-post state, never partial (`readers_keep_searching_during_compaction`);
   concurrent writes are never lost — seq-checked swap with off-lock retry and
   an under-lock fallback (`concurrent_write_during_compaction_is_preserved`).
5. ✅ Streaming-CRC corruption of any segment byte fails open with an
   actionable `Corrupt` error (`segment::tests`,
   `corrupt_snapshot_is_detected_not_silently_loaded`,
   `corrupt_delta_segment_is_detected_not_silently_loaded`).
6. ✅ v1→v2 migration (`v1_vault_migrates_to_v2_on_reopen_and_flush`,
   `v2_flush_writes_binary_segment`).
7. ✅ `differential_consistency.rs` extended with a multi-segment config;
   fmt/clippy clean.

## Validação

**All acceptance criteria are met.** The delta flush is O(delta); a bounded
delta budget and `compact()` keep reopen cost bounded; compaction rebuilds
off-lock so readers are never starved. Storage v2 is complete per this ADR.
Future niceties (background-scheduled compaction, delta-segment mmap) would be
new ADRs, gated on measured need.
