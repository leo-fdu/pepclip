# Real encoder extraction and isolated GPU stages, 2026-10-26

On the rented RTX 4090, precomputed inputs allow ESM2-650M and GROVER-base to be extracted in approximately **22.5 hours for ten million pairs, or 27.3 hours for all 12,124,168 pairs**, using AMP FP16 and running the two encoders sequentially. These are short-sample throughput extrapolations of the GPU stage with prepared CPU inputs, not full-data execution times or end-to-end rental-duration guarantees.

## GPU stage with CPU preparation completed

| Encoder | Precision | Batch | Samples/s | 10 million | 12,124,168 | Peak allocated / reserved GiB |
|---|---|---:|---:|---:|---:|---:|
| ESM2-650M | fp32 | 32 | 228.8 | 12.1 h | 14.7 h | 2.80 / 3.01 |
| ESM2-650M | amp_fp16 | 32 | 586.5 | 4.7 h | 5.7 h | 2.67 / 2.84 |
| GROVER-base | fp32 | 4 | 108.7 | 25.6 h | 31.0 h | 0.88 / 2.71 |
| GROVER-base | amp_fp16 | 16 | 156.5 | 17.8 h | 21.5 h | 1.42 / 3.17 |

Prepared-input timing includes CPU→GPU copies, the real frozen model forward, the configured pooling, and CPU FP32 output transfer. Input preparation, NPZ decoding, disk reading/writing, network transfer, weight acquisition/loading, and full-cache verification are outside this stage. A second control preloads inputs into GPU memory and leaves outputs there; its rates are close to the prepared-CPU cases. CUDA event and wall-clock times are both recorded. GPU-resident rates still include software launch/stream gaps and are not a sum of idealized kernel compute times.

Models and pooling are unchanged: ESM2 esm2_t33_650M_UR50D uses residue means excluding special/padding tokens; GROVER-base uses the concatenation of atom_from_atom and atom_from_bond molecule means, with no handcrafted descriptors or finetuning heads. Backbones are frozen and in eval mode. TF32 is disabled. AMP FP16 is a measured inference option and is not silently enabled in the normal extract CLI. Eight-sample checks gave mean cosine similarity to FP32 of 0.9999997 for ESM2 and approximately 1.0 for GROVER, with max absolute differences 0.00747 and 0.000940 respectively; this does not establish downstream equivalence.

## Full extraction path before CPU offload

The unchanged serial CPU graph path runs at only about 7–8 peptides/s. Across configurations, graph construction and BatchMolGraph assembly occupy about 91–95% of measured extraction time. The upstream MolGraph scans atom pairs when discovering bonds; GPU capacity does not eliminate that Python/CPU work.

Including the separately measured sequence-to-SMILES preparation rate (204.6/s), the fastest measured GROVER end-to-end option extrapolates to 15.6 days for ten million or 18.9 days for the full set. These much longer numbers apply to serial CPU preparation on the cloud VM, not to GPU work after preprocessing.

ESM2's ordinary adapter path, including tokenization, CPU return, finite checks, and memmap write/flush, measured 202.5/s in FP32 and 429.2/s with AMP FP16: approximately 13.7/6.5 hours for ten million, or 16.6/7.9 hours for the full set. Differences from the pretokenized GPU table are part of the software/input path rather than a different model.

## CPU-prepared graph storage

A representative 256-peptide input set, packed in batches of 16, required 305,057,712 uncompressed bytes and 5,069,510 bytes with lossless NumPy NPZ compression. This extrapolates to roughly **11.9 TB raw / 198 GB compressed for ten million**, or **14.4 TB raw / 240 GB compressed for the full set** (decimal units). These are graph-input caches, separate from the roughly 140 GB of final dual-side FP32 embedding matrices.

All cached arrays survived compression/decompression bitwise exactly; the first restored graph batch also reproduced the FP32 model outputs within rtol/atol 1e-5. The sample took 3.15 s to compress and 0.666 s to decode on the cloud host. This does not measure Mac preprocessing speed. A production pipeline still needs sample-ID indices, provenance, sharding, checksums, resumption, and asynchronous decoding/prefetch; benchmark support alone is not a completed full-scale preparation/extraction pipeline. Full CPU preparation before GPU rental is feasible with enough storage, or prepared shards can be staged progressively.

## Sampling, environment, and validation

Uniform reservoir sampling (seed 42) selected 2048 actual sequences from the complete filtered FASTA; sample mean length is 75.10 aa versus population 74.62 aa. The timed end-to-end windows comprise 1536 ESM2 sequences (mean 74.98 aa) and 192 GROVER sequences (mean 74.85 aa); GPU-stage tests use 512 and 256 respectively. Each configuration has three timed repeats after two warmups. Fixed prepared-input sets are reused for GPU timing. These samples do not quantify every long-run or rare-composition effect.

GROVER sample molecules have about 589 heavy atoms on average and use the project's standardized linear L-peptide/free-termini graph policy. They are not asserted to be the native mature structures. These benchmarks do not validate downstream generalization or cyclic/noncanonical peptide performance.

Hardware: RTX 4090 24,564 MiB, driver 550.142; Ubuntu 22.04, Python 3.10.12, four CPU threads, approximately 31 GiB RAM and 10 logical CPUs. Software and source hashes are in [provenance](20261026-encoder-provenance.json). GROVER source is official commit 40b6d97098e4508687912f3c05eca369fc2c6213. ESM weights came from the official FAIR URL. Official GROVER download links were unavailable from this host; its original-format checkpoint came from the documented public mirror and matched that mirror's pinned SHA256. It was not independently byte-compared against the official Drive copy.

Real loading uncovered missing inference-only fields in the encoder-only GROVER checkpoint and namespace-package source discovery. The adapter now defaults omitted dropout to 0 (eval disables dropout), initializes only the missing nonlearned zero readout sentinel, and records namespace source hashes correctly. Every learned parameter and every supplied required tensor shape still must match; the original checkpoint bytes are unchanged. Three adapter regression tests passed on the cloud host, including rejection of missing learned weights and wrong sentinel shapes.

All 20 end-to-end and 32 isolated-stage configurations completed with real pretrained weights. Generated temporary output matrices were removed; source, weights, logs, and result JSON remain in the dedicated server directory. No full-data job was started. All benchmark jobs finished and the GPU returned to idle.

## Reproduction

From the repository checkout on the validated CUDA environment:

```bash
export PYTHONPATH=src:/path/to/grover-checkout
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# fair-esm 2.0.0 uses legacy torch.load for its trusted official checkpoint:
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python scripts/benchmark_encoders.py --model esm2 \
  --checkpoint /path/to/esm2_t33_650M_UR50D.pt --sample /path/to/sample.json \
  --output outputs/esm2-extraction.json --batch-sizes 32 64 128 256 512 \
  --samples-per-repeat 512 --repeats 3

python scripts/benchmark_encoders.py --model grover \
  --checkpoint /path/to/grover_base.pt --sample /path/to/sample.json \
  --output outputs/grover-extraction.json --batch-sizes 4 8 16 32 64 \
  --samples-per-repeat 64 --repeats 3

python scripts/benchmark_encoder_gpu.py --model grover \
  --checkpoint /path/to/grover_base.pt --sample /path/to/sample.json \
  --output outputs/grover-gpu.json --batch-sizes 4 16 64 --samples 256

python scripts/benchmark_encoder_gpu.py --model esm2 \
  --checkpoint /path/to/esm2_t33_650M_UR50D.pt --sample /path/to/sample.json \
  --output outputs/esm2-gpu.json --batch-sizes 32 64 128 256 512 --samples 512
```

The regression sample is derived, ignored data. Reproduce it by reading the filtered FASTA with read_fasta, retaining the first 2048 rows, and using random.Random(42).randrange(i+1) reservoir replacement for each later zero-based index i; rows contain sample_id=header.split()[0] and sequence. The source filtered FASTA SHA256 is dac764132448f462a09c2eef807797464e1d01757fbacfca3c1a876c14aea8d2; every result records the sample SHA256. Keep the original source/output provenance when reproducing.

Raw measurements: [ESM2 full path](20261026-rtx4090-esm2-extraction.json), [GROVER full path](20261026-rtx4090-grover-extraction.json), [ESM2 GPU stages](20261026-rtx4090-esm2-gpu-stage.json), [GROVER GPU stages](20261026-rtx4090-grover-gpu-stage.json).
