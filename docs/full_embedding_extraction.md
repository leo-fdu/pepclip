# Full ESM-2 3B / GROVER base extraction

The 2026-10-27 run targets all 12,124,168 retained UniRef90 representatives,
10–100 canonical uppercase amino acids inclusive. The source release is unknown.
Its filtered FASTA SHA256 is
`dac764132448f462a09c2eef807797464e1d01757fbacfca3c1a876c14aea8d2`.
Extraction retains unique source IDs and source row order; it assigns no split.
Deduplicate identical sequences when selecting training pairs. These graphs are
standardized linear L-peptides with free termini and no extra crosslinks,
cyclization, or modifications; they do not establish native chemical structures.

## Runtime and weights

The reference settings are in
[`configs/extraction_esm3b_grover_base.json`](../configs/extraction_esm3b_grover_base.json).
This file documents settings; the explicit extraction CLI and `run.json` are the
authoritative executed configuration. The target is Linux, Python 3.10.12,
PyTorch 2.6.0+cu124, four RTX 4090 GPUs (23,028 MiB each), driver 595.58.03,
44 allocated vCPUs and 125 GiB RAM. Guest CPU model is AMD EPYC 7402;
allocation does not prove dedicated physical cores or host SMT topology.
Use the data disk for the environment, weights, inputs, spool and outputs.

```bash
python3.10 -m venv "$RUN_DATA/venv"
"$RUN_DATA/venv/bin/pip" install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$RUN_DATA/venv/bin/pip" install numpy==2.2.6 fair-esm==2.0.0 rdkit==2025.9.4 \
  scipy==1.15.3 pandas==2.2.3 scikit-learn==1.6.1
"$RUN_DATA/venv/bin/pip" freeze > "$RUN_DATA/requirements-frozen.txt"
```

Acquire both the main ESM weight and its auxiliary regression checkpoint from
FAIR; the local ESM loader expects both files together, although contact
predictions are disabled:

- `https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt`
- `https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt`

GROVER source is pinned to `40b6d97098e4508687912f3c05eca369fc2c6213` of
`tencent-ailab/grover`. Base weights came from the public original-format mirror
`https://github.com/OlivierBeq/jax-grover/releases/download/model_weights/grover_base.pt`.
Verify SHA256
`47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de`.
This verifies the mirror pin, not an independent byte comparison to official
Drive weights. The GROVER adapter explicitly opts into loading this trusted
legacy checkpoint. ESM allowlists only `argparse.Namespace` for safe loading on
PyTorch 2.6; it does not disable the safe loader.

## Execution and recovery

Set `RUN_DATA`, `CHECKOUT`, `GROVER_SOURCE`, `FILTERED_FASTA` and `SOURCE_REVISION`
to the target paths and the deployed source commit. No model runs on the Mac.

```bash
cd "$CHECKOUT"
export PYTHONPATH="$CHECKOUT/src:$GROVER_SOURCE"
"$RUN_DATA/venv/bin/python" -u scripts/extract_embeddings.py \
  --fasta "$FILTERED_FASTA" \
  --fasta-sha256 dac764132448f462a09c2eef807797464e1d01757fbacfca3c1a876c14aea8d2 \
  --expected-count 12124168 --output "$RUN_DATA/results/esm3b-grover-base" \
  --grover-source "$GROVER_SOURCE" \
  --esm-checkpoint "$RUN_DATA/weights/esm2_t36_3B_UR50D.pt" \
  --grover-checkpoint "$RUN_DATA/weights/grover_base.pt" \
  --git-revision "$SOURCE_REVISION" --cpu-workers 40 --gpu-cpu-threads 1 \
  --gpus 4 --esm-gpus 2 --esm-batch 32 --grover-batch 8 \
  --shard-size 4096 --spool-gb 20
```

Both models are frozen and in eval mode. FP32 parameters are executed under
FP16 autocast; final pooled, unnormalized vectors are FP32. ESM uses final-layer
residue mean excluding BOS/EOS/padding (2560 dimensions). GROVER concatenates
the two upstream mean-atom readouts (1600 dimensions). Approximate combined
feature storage is 202 GB, plus IDs/metadata and a bounded graph spool.

Forty single-thread CPU processes construct graphs in parallel. `fast_graph.py`
is opt-in and refuses an unknown upstream file checksum. It preserves sorted
atom-pair bond order and every original feature, skipping only absent bonds.
Every packed batch is restored and checked before spool publication. GPU
workers prefer two cards per model initially, then claim the other model's
available work. Length sorting is restricted to execution inside a shard;
vectors are restored to FASTA row order before output. Graph decoding overlaps
CUDA inference. A bounded spool applies backpressure to CPU producers.
Utilization depends on the bottleneck; this does not manufacture CPU work or
allocate extra memory merely to inflate utilization statistics.

Each committed shard has a row interval, SHA256, GPU, timing and peak allocated
memory. Output writes are flushed before an atomic progress marker is written.
Incomplete shards are recomputed after restart; committed shards are checked
again during finalization. Only one supervisor may own a run. After a hard OS
kill, ensure no old child processes remain before restarting the same command.
Source/configuration changes are rejected on resume. Do not edit a deployed
run's files or change its extraction policy in place.

`status.json` and `telemetry/history.jsonl` record GPU utilization, memory,
power/temperature, CPU interval utilization/iowait/steal, process resource use,
free disk and completed shard counts every 30 seconds. A worker failure stops
the run and preserves committed output. Finalization checks every shard and
full-file hashes; each completed model becomes available for transfer before
the other finishes. `complete.json` marks both cloud bundles finished.

## Local return

Run the receiver on the Mac, keeping it awake and connected:

```bash
python3 -u scripts/receive_embeddings.py --host "$CLOUD_HOST" --user ubuntu \
  --remote "$RUN_DATA/results/esm3b-grover-base" \
  --known-hosts "$TASK_KNOWN_HOSTS" \
  --output embeddings/uniref90-esm3b-grover-base-20261027
```

The receiver asks for the password once and keeps it only in process memory.
It polls every two minutes, downloads completed bundles with resumable rsync,
checks feature/ID SHA256 values, and records `download-verified.json` per model.
Local `complete.json` means both bundles passed local verification. It creates
no server login keys or saved password file. A sleeping/offline Mac delays
return; restart the same receiver command to resume. Network speed can make
return longer than inference. No cloud instance is stopped or deleted by these
scripts; retain cloud output until the local hashes pass.

### Overlap return with extraction

The live run additionally stages only committed row ranges and transfers them
while the encoders continue. This preserves the exact original FP32 bytes and
NPY headers, with no re-encoding or quantization. Start the read-only stager:

```bash
"$RUN_DATA/venv/bin/python" -u scripts/stream_embedding_return.py stage \
  --root "$RUN_DATA/results/esm3b-grover-base" --staging "$RUN_DATA/transfer-spool"
```

On the Mac, use this receiver instead of the whole-bundle receiver above:

```bash
python3 -u scripts/stream_embedding_return.py receive --host "$CLOUD_HOST" \
  --remote-staging "$RUN_DATA/transfer-spool" --known-hosts "$TASK_KNOWN_HOSTS" \
  --output embeddings/uniref90-esm3b-grover-base-20261027
```

Each staged chunk is checked against its original extraction progress SHA256.
The receiver verifies chunks, writes disjoint rows into sparse NPY files, flushes
them, and commits local receipt markers. The final local file hashes must equal
the cloud bundle metadata before `download-verified.json` and `complete.json`
are published. An incomplete NPY has its final apparent size from the start;
use `receive-status.json` / received shard counts to measure actual progress.
No local NumPy, model weights, or encoder runtime is required. A process lock
prevents duplicate stagers/streaming receivers for the same directory.

Staging can temporarily duplicate up to 202 GB on the cloud disk; the 500 GiB
disk has space for these chunks, the full matrices, the bounded graph spool,
weights, environment and inputs. The local transfer cache contains a small
batch of chunks and each chunk is removed after assembly. Keep cloud staging
until both final local hashes pass. Both original and streamed return paths
passed the real 512-sample ESM3B/GROVER test, including exact final NPY SHA256
matches. A standard-library test checks out-of-order assembly, resumption,
unchanged headers and rejection of a different source run.

The live graph workers were reniced from 0 to 5 to prioritize GPU launch/decode
threads while retaining all 40 producers; the operation is recorded in cloud
`runtime-tuning.jsonl`. This changes scheduling, not features or precision.
Restarted workers use their normal priority unless this operational tuning is
reapplied.

## Preflight evidence

On this target, 2,048 sampled peptides had bitwise identical eight-component
graphs with the original/optimized builders and lossless codec. Summed graph
construction CPU times improved about 2.32x; this is sampled, not full-run
throughput. Real GROVER outputs on eight samples were exactly equal between
original and optimized/restored inputs. Three adapter tests passed.

A short 256-sample test measured ESM 3B batch 32 at 150.30 samples/s and GROVER
base batch 8 at 165.14 samples/s. Larger batches were slower in these tests.
ESM AMP-vs-FP32 minimum cosine on eight samples was 0.99999547, max absolute
difference 0.00829434; GROVER was 0.99999994 / 0.00093973. These are numerical
checks, not downstream accuracy validation. Full-run throughput, utilization,
and successful local return must be established by the actual run artifacts.

The four-GPU 512-sequence end-to-end preflight completed both schema-2 bundles;
the regular embedding loader accepted both. Repeating the identical command
recognized the completed run without recomputation. Both small bundles were
automatically returned to the Mac and passed SHA256 verification. Detailed
preflight model/source hashes and measurements are in
[`benchmarks/20261027-four4090-preflight.json`](benchmarks/20261027-four4090-preflight.json).
