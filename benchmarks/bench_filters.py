#!/usr/bin/env python3
"""Typed-filter selectivity benchmark: 20k docs, dense query with metadata
filters at 100% / 50% / 10% / 1% / 0.1% selectivity.

Two paths with IDENTICAL semantics per selectivity:
- typed:     {"bucket": value}                -> typed-index prefilter
- predicate: {"$or": [{"bucket": v}, {"bucket": v}]} -> not extractable,
             per-candidate JSON predicate (the pre-index behavior)

Every number is measured on this machine; writes RESULTS-FILTERS.md.
"""
import json, platform, statistics, tempfile, time
from pathlib import Path
import numpy as np
from ragvault import _native

N, DIM, K, QUERIES = 20_000, 64, 10, 60
rng = np.random.default_rng(7)
data = rng.standard_normal((N, DIM), dtype=np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)

tmp = tempfile.mkdtemp(prefix="rv-filt-")
config = {"dim": DIM, "metric": "cosine",
          "hnsw": {"m": 16, "ef_construction": 200, "ef_search": 64, "seed": 7},
          "bm25": {"k1": 1.2, "b": 0.75, "lowercase": True},
          "wal_sync": "batch", "flat_threshold": 10}
vault = _native.Vault.open(tmp, json.dumps(config))
# selectivity buckets: all=100%, half=50%, dec=10%, cent=1%, mil=0.1%
t0 = time.monotonic()
for start in range(0, N, 500):
    rows = data[start:start + 500]
    doc_id = f"doc-{start}"
    chunks = []
    for i in range(len(rows)):
        g = start + i
        chunks.append({
            "chunk_id": f"{doc_id}#{i}", "document_id": doc_id,
            "document_version": 1, "chunk_index": i, "text": "",
            "section_path": [],
            "metadata": {"all": "yes",
                         "half": "h" + str(g % 2),
                         "dec": "d" + str(g % 10),
                         "cent": "c" + str(g % 100),
                         "mil": "m" + str(g % 1000)}})
    vault.upsert_document(json.dumps({"document_id": doc_id,
                                      "current_version": 1, "metadata": {}}),
                          json.dumps(chunks), np.ascontiguousarray(rows))
print(f"ingest: {time.monotonic()-t0:.1f}s")

def measure(filt):
    lat = []
    for qi in range(QUERIES):
        q = data[(qi * 331) % N]
        req = {"k": K, "mode": "dense", "candidates": K, "filter": filt}
        t = time.monotonic()
        r = vault.search(json.dumps(req), np.ascontiguousarray(q))
        lat.append((time.monotonic() - t) * 1000)
    lat.sort()
    return statistics.mean(lat), lat[len(lat)//2], lat[int(len(lat)*0.95)], r["plan"]

cases = [("100%", {"all": "yes"}), ("50%", {"half": "h1"}),
         ("10%", {"dec": "d3"}), ("1%", {"cent": "c42"}),
         ("0.1%", {"mil": "m123"})]
lines = ["# Typed-filter selectivity benchmark",
         "",
         f"- {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, "
         f"{platform.platform()}, n={N:,}, dim={DIM}, k={K}, {QUERIES} queries, "
         "dense over HNSW-eligible collection",
         "",
         "| selectivity | typed prefilter p50/p95 ms | backend | "
         "predicate-path p50/p95 ms | backend | speedup (p50) |",
         "|---|---|---|---|---|---|"]
for name, filt in cases:
    key, val = next(iter(filt.items()))
    _, tp50, tp95, tplan = measure(filt)
    _, pp50, pp95, pplan = measure({"$or": [{key: val}, {key: val}]})
    lines.append(f"| {name} | {tp50:.2f} / {tp95:.2f} | "
                 f"{tplan['dense_backend']} | {pp50:.2f} / {pp95:.2f} | "
                 f"{pplan['dense_backend']} | {pp50/max(tp50,1e-9):.1f}x |")
    print(lines[-1])
lines += ["", "> Both columns run semantically identical filters; the "
          "predicate column uses a non-extractable shape ($or of the same "
          "clause) to force per-candidate JSON evaluation. Typed path "
          "switches to bitmap_prefiltered_flat when the candidate set is "
          "small; selectivity and coverage are reported in every plan."]
Path("benchmarks/RESULTS-FILTERS.md").write_text("\n".join(lines) + "\n")
print("written: benchmarks/RESULTS-FILTERS.md")
import shutil; vault.close(); shutil.rmtree(tmp, ignore_errors=True)
