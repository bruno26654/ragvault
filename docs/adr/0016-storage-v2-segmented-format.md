# ADR 0016 — Storage v2: segmented binary format

Status: **in progress** (scopes the v0.2 work deferred by ADR 0004).
Supersedes ADR 0004's "single mutable arena" decision when fully built.

**Implemented so far:** the binary segment container (`crate::segment`:
length-prefixed records, per-record CRC, streaming footer CRC, incremental
verification) and the v2 base format — `format_version = 2` writes the base
state as `gen-N/state.rvseg` instead of `state.json`, with transparent v1→v2
migration on reopen. **Remaining:** multi-segment delta flush (O(delta)),
read-safe online compaction, and the read-during-compaction concurrency
guarantee (bullets 1–5 below, partially met; see the acceptance list).

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

1. ◻️ Round-trip across ≥2 delta segments (differential harness extended).
   *(Single-segment round-trip via `state.rvseg` is ✅, covered by every
   reopen test; multi-segment deltas are not built yet.)*
2. ◻️ Tombstone/version shadowing correct across segment boundaries.
3. ◻️ Crash injection between (a) segment fsync and manifest rename,
   (b) manifest rename and old-segment GC — reopen consistent both times.
   *(The manifest rename + fsync + GC-after-durable protocol from ADR 0004 is
   ✅ and unchanged; the multi-segment variant is not built.)*
4. ◻️ Read-during-compaction: a reader on a pinned snapshot returns stable
   results while a compaction publishes and GCs.
5. ✅ Streaming-CRC corruption of any segment byte fails open with an
   actionable `Corrupt` error (`segment::tests`,
   `corrupt_snapshot_is_detected_not_silently_loaded`).
6. ✅ v1→v2 migration: a v1 vault opens and first flush produces a valid v2
   base (`v1_vault_migrates_to_v2_on_reopen_and_flush`,
   `v2_flush_writes_binary_segment`).
7. ◻️ `differential_consistency.rs` extended with a multi-segment config;
   fmt/clippy clean (✅ for what exists).

## Validação

Partially implemented. The binary base format (v2) + migration + streaming-CRC
container are done and tested. The multi-segment delta path, O(delta) flush,
and read-safe online compaction remain — until those land the engine keeps the
single-base-per-generation model (ADR 0004's publish protocol, now writing a
binary base). `TASKS.md` tracks the remaining units.
