# M5 Pro offline CPU preparation benchmark, 2026-10-26

**All CPU preparation can precede GPU rental**, provided sufficient prepared-input storage is available and the shard producer/consumer are connected. On this Mac, use roughly **one day, with 1–1.5 days of planning allowance**, rather than the optimistic short-run maximum. No full preprocessing job was started.

Measured on Apple M5 Pro, 15 CPU cores, 24 GiB unified memory, macOS 27.0.1, AC power. The user explicitly requested this local CPU benchmark; an isolated environment was created at .venv/cpu-benchmark without changing the existing environment's installed packages or the system Python. No model weights were downloaded or loaded on the Mac, and MPS/CUDA were not used.

## CPU time

GROVER preparation includes sequence-to-standard-linear-L-peptide SMILES, unchanged upstream MolGraph features, BatchMolGraph assembly, lossless NPZ compression, write/fsync, exact decompression/array roundtrip, process IPC, and scheduling. Sixteen peptides form a cache chunk. Each worker uses one library thread. Spawn/import startup is recorded separately and excluded from steady rates. The short scaling run used the same 512 real sampled peptides twice per worker configuration:

| Processes | Short-run peptides/s | Full 12,124,168 extrapolation |
|---:|---:|---:|
| 1 | 26.2 | 128.7 h |
| 2 | 49.4 | 68.2 h |
| 4 | 91.0 | 37.0 h |
| 8 | 157.6 | 21.4 h |
| 12 | 202.0 | 16.7 h |

The 12-process sustained run repeated the complete 2048-peptide reservoir sample 16 times. Its median was 187.8/s, but throughput declined from 225.1 to 139.3/s, so that median alone is an unsuitable long-run scheduling promise. Thermal telemetry was unavailable and the cause of the drift was not established; memory pressure query reported 41% system-wide free memory.

An ensuing already-warm 8-process run repeated 2048 peptides eight times. Median throughput was **154.8/s**, with the final four repeats near 152–155/s. This corresponds to **17.9 hours for ten million**, or **21.8 hours for all retained peptides**. This is a minutes-long load test, not an overnight stability validation. Sleep, other workloads, disk behavior, and long-run variation remain outside the extrapolation.

ESM2 token preparation is much smaller: ESM-1b alphabet from fair-esm 2.0.0 (the ESM2 alphabet), batch 32, int64 token/length arrays, NPZ compression/fsync, and exact roundtrip. Five runs over 2048 peptides measured **4796/s**, or **0.70 hours (~42 minutes)** for the full set. No ESM2 model is needed to tokenize. Serially adding graph preparation and token preparation gives approximately **22.5 hours** before other operational overhead.

These measurements do not establish the speed of an untested CPU server. If a roughly one-day local run is acceptable, CPU rental is not required by compute capacity alone. If completion within a few hours or avoiding Mac occupation matters, benchmark the proposed server with the same script; vCPU count alone is not evidence of proportional speedup.

## Storage

At the measured full-reservoir compression rate:

| Prepared input | Ten million | Full 12,124,168 |
|---|---:|---:|
| GROVER lossless graph NPZ | 198.7 GB | 241.0 GB |
| ESM2 lossless token NPZ | 0.95 GB | 1.15 GB |
| Combined | 199.7 GB | 242.1 GB |

Units are decimal GB. File/index metadata, temporary files, safety margin, input FASTA, model weights, and final embedding caches are additional. Early disk readings were about 209 GB available; a later reading was 231.4 GB as system availability changed. Neither safely accommodates the estimated complete prepared cache plus margin. Plan at least **300 GB available space** for this phase. Retaining final FP32 embeddings as well adds approximately 140 GB, before import copies.

Using existing storage alone requires staged processing/upload with bounded local buffers, which differs from keeping every CPU result locally before beginning GPU work. A larger/external volume or remote/object storage allows the requested CPU-first schedule. Compression preserves arrays; replacing prepared graph features with SMILES alone would leave the costly graph feature work on the GPU host.

## Correctness and provenance

All written NPZ chunks were decoded and their arrays checked bitwise. The first 16-peptide graph chunk matched the cloud host's prepared input byte-for-byte across macOS/Python3.14/Torch2.10/NumPy2.4 and Ubuntu/Python3.10/Torch2.6/NumPy2.2, using RDKit 2025.9.4 and identical official GROVER source. Digest: 6b8488451deff2a30de411d3a2aa6446d3f5fdcc06b6a16dca92b56ee7e30cf7. The cloud GPU benchmark separately verified a restored graph batch's model outputs. This is sample validation, not proof for every future input or a completed shard consumer implementation.

GROVER source is pinned to 40b6d97098e4508687912f3c05eca369fc2c6213. Full environment and code hashes: [CPU provenance](20261026-m5pro-cpu-provenance.json). Three adapter regression tests also passed locally after measurement. Generated environments, upstream source, real sampled rows, graph shards, indices, and logs remain ignored; only selected summaries and scripts are version-controlled.

## Reproduction

Use the explicitly separate CPU-only profile; it is not the cloud encoder environment:

```bash
python3.14 -m venv .venv/cpu-benchmark
.venv/cpu-benchmark/bin/python -m pip install -r requirements/cpu-benchmark-macos.txt
export PYTHONPATH=src
.venv/cpu-benchmark/bin/python scripts/benchmark_cpu_graphs.py \
  --grover-source /path/to/pinned-grover-checkout --sample /path/to/sample.json \
  --output outputs/cpu-graph-scaling/results.json --workers 1 2 4 8 12 --samples 512 --repeats 2
.venv/cpu-benchmark/bin/python scripts/benchmark_cpu_graphs.py \
  --grover-source /path/to/pinned-grover-checkout --sample /path/to/sample.json \
  --output outputs/cpu-graph-sustained/results.json --workers 8 --samples 2048 --repeats 8
.venv/cpu-benchmark/bin/python scripts/benchmark_cpu_tokens.py \
  --sample /path/to/sample.json --output outputs/cpu-tokens/results.json
```

These are bounded benchmark entry points. A full-data producer still needs streaming input, durable ID/provenance indices, bounded-size shard directories/archives, atomic completion and resumption, disk-space guards, and a matching GPU shard reader. Keep chemical preprocessing, model pooling, and ID pairing fixed. No full-data cache or model-training run has been launched.

Raw summaries: [scaling](20261026-m5pro-cpu-scaling.json), [12-process sustained](20261026-m5pro-cpu-sustained12.json), [8-process sustained](20261026-m5pro-cpu-sustained8.json), [ESM tokens](20261026-m5pro-cpu-tokens.json). See [cloud encoder timing](20261026-encoder-extraction.md) for the separately measured ~27.3 hours of full-set GPU work.
