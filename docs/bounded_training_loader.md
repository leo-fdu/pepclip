# Bounded training batch reads

The optional `batch_reads=1` path replaces per-sample SQL queries and tensor
construction with chunked batch queries and two preallocated float32 tensors.
Feature positions are sorted only for reading adjacent rows together; tensors
are restored to the exact sampler order. Pairing, split, shuffle, batch size,
precision, loss, optimizer and per-step nonfinite checks are unchanged.

`prefetch_batches=1` prepares at most one additional batch on one background
thread. Its read-only SQLite connection and independent file handles are
created, used and closed on that thread. The executor joins pending work on
normal completion, early consumer exit and exceptions. No feature array is
mapped for training or loaded in full. Validation retains its original loader.

These switches default to zero. They and `cpu_threads` may change on resume;
scientific configuration and input provenance checks still apply.
`configs/precomputed_mps_resume_30epochs.json` enables both switches and sets
`cpu_threads=10`, with PyTorch inter-op threads set to one. Together with the
single reader this leaves nominal CPU capacity on the 15-core development Mac;
this is not CPU affinity or a guarantee of desktop responsiveness. At launch,
also set `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=10` to bound native library pools before
Python imports. Use the original environment (Python 3.10.19, PyTorch 2.10.0,
NumPy 2.2.6); the total-epoch setting remains 30 and resume source is epoch 20
`last.pt`. Starting a full training run is a separate action.

At batch 512, the two 2560/1600-dimensional feature tensors occupy about
8.125 MiB per batch. Prefetch adds bounded batch buffers, an 8 MiB SQLite cache,
and transient read buffers; it does not increase model batch size or GPU
activation sizes. This bounds loader memory, but cannot guarantee no OOM from
the full model, checkpoint serialization or competing applications.

## Validation

In the existing original environment, `tests/test_low_memory.py` covers exact
batch values/order, repeated indices, SQL chunking, early exit, read exceptions,
thread cleanup, and exact small-model weights for ordinary versus optimized
training and resume. No full MPS training was launched for this change.

A short I/O-only probe read real feature rows using a synthetic 65,536-row
identity index (not actual training pairs). For 32 batches of 512, warm-cache
legacy reads took 0.211 s; batched reads took 0.142–0.161 s; prefetched reads
0.140–0.141 s. The first legacy pass took 2.782 s and is not a fair warm-cache
comparison. Whole probe process peak RSS was 352.6 MiB. These measurements do
not establish end-to-end training speed or full training peak memory.
