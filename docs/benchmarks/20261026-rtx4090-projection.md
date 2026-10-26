# RTX 4090 projection-only benchmark, 2026-10-26

Measured on a dedicated NVIDIA RTX 4090 (24,564 MiB, 450 W power limit), driver 550.142, Ubuntu 22.04, Python 3.10.12, PyTorch 2.6.0+cu124, CUDA runtime 12.4, and NumPy 2.2.6. The host exposes 10 logical CPUs and about 31 GiB RAM; the benchmark uses four CPU threads and the existing single-process DataLoader. No ESM2/GROVER model weights or real peptide data were uploaded.

The exact project model and loss were used: two independent Linear–GELU–Linear heads, input dimensions 1280/1600, hidden width 512, projected dimension 256, L2 normalization, fixed temperature 0.07, and bidirectional batch-local cross-entropy. There are 1,738,240 trainable parameters; only the heads use AdamW (learning rate 0.001, weight decay 0.0001). Synthetic inputs have no biological accuracy interpretation.

Precision is FP32 with TF32 and AMP disabled; deterministic algorithms are enabled with CUBLAS_WORKSPACE_CONFIG=:4096:8. Each case uses 10 warmup steps and three separately timed repeats of 20 complete forward/backward/optimizer steps, including the trainer's finite-loss check and scalar synchronization. Rates below are medians. Peaks include warmup and optimizer-state initialization.

| Batch | GPU-resident pairs/s | Warm-cache loader pairs/s | Peak allocated GiB | Peak reserved GiB |
|---:|---:|---:|---:|---:|
| 2,048 | 488,203 | 44,414 | 0.19 | 0.20 |
| 4,096 | 800,345 | 41,330 | 0.42 | 0.48 |
| 8,192 | 614,315 | 37,182 | 1.26 | 1.28 |
| 16,384 | 349,948 | 35,925 | 4.45 | 5.45 |
| 32,768 | 177,559 | 32,816 | 16.81 | 20.81 |

Both modes successfully ran all five batches. Allocated/reserved peaks match between modes at this precision and shape. Reserved memory is the CUDA allocator's retained pool, not an additional allocation to add to allocated memory; neither figure includes all CUDA context/driver memory.

The resident case reuses one already-resident GPU input batch and excludes transfer/data loading. The loader case uses the actual PairedFeatures and ContrastiveBatches path over 65,536 synthetic pairs in memory-mapped .npy files, with num_workers=0, unpinned CPU tensors, and blocking CPU→GPU copies. Its approximately 0.75 GB matrix data fits in host RAM and benefits from warm OS page cache. It does not measure cold SSD reads or random access over the approximately 140 GB full-scale cache.

These measurements support using 4090 for projection training and show substantial overhead in the current per-sample Python/cache/transfer path. Batch 32768 fits but is slower in pairs/s and leaves less memory margin. Batch 4096/8192 remains a reasonable performance experiment; do not select batch only by throughput, because negative-pool size and optimization steps also change. No training batch default or data split was changed by this benchmark.

Do not interpret these rates as ESM2/GROVER embedding extraction throughput, Mac MPS/CPU performance, downstream accuracy, or a measured full-data epoch duration. The VM had about 62 GB free disk before setup, so it cannot currently hold the estimated full dual-side FP32 cache. Framework installation/network time is excluded from the benchmark rates.

## Validation and reproduction

All six tests in tests/test_embeddings.py passed on this target environment (0.940 s), covering schema-2 pairing/import validation, gradients, batch-local validation, CLI training/evaluation, exact CPU resumption, and projection-only inference. These tests use synthetic vectors; neither real encoder extraction nor the chemistry-dependent suite was run.

The main source revision was 0ba3150; the raw result records SHA256 of model.py, cache.py, and train.py to identify the exact benchmarked implementations. The benchmark script and result are committed together. The source was copied into a dedicated server-side checkout and dependencies installed in an isolated virtual environment. PyTorch's CUDA packages use the 12.4 build; NumPy and the PyPI PyTorch wheel were obtained through the Tsinghua PyPI mirror after slow public download routes.

On a compatible CUDA 12.4 target with Python 3.10, torch==2.6.0 and numpy==2.2.6 installed, run:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m unittest discover -s tests -p test_embeddings.py -v
python scripts/benchmark_projection.py \
  --output outputs/benchmarks/projection-4090.json \
  --warmup 10 --steps 20 --repeats 3
```

The script refuses an existing result file/cache directory, creates only synthetic caches beside the specified result, records each case immediately, and records CUDA out-of-memory cases without silently shrinking the batch. The cloud test was bounded by a 1200-second process timeout; the job completed normally and the GPU returned to idle (1 MiB used). No long training job was left running.

Raw machine-readable measurements: [20261026-rtx4090-projection.json](20261026-rtx4090-projection.json).
