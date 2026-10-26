# Storage and upload probe, 2026-10-26

The Mac's newly measured available space was 299.83 decimal GB (279.24 GiB). The estimated complete graph/token prepared cache is about 242 GB, leaving about 58 GB before indices, filesystem overhead, and temporary files. This fits the CPU-only phase with limited margin; retaining another approximately 140 GB of final dual-side FP32 embeddings simultaneously does not fit. Available storage is a live snapshot, not a fixed machine specification.

A 67,108,864-byte (64 MiB) random synthetic file was uploaded with native SCP to the same user-authorized cloud host. SCP completed successfully, displayed an average 4.1 MB/s and approximately 15 seconds, after the connection's initial ramp-up. Compression was not enabled. Authentication/connection waiting is outside SCP's file-transfer meter. This is a single short upload, not a sustained bandwidth guarantee or a download-speed measurement.

For planning, 242 GB / 4,000,000 bytes/s is about 16.8 hours of upload alone. At 2 MB/s it is 33.6 hours; at 10 MB/s it is 6.7 hours. Network fluctuations, reconnects, long-run limits, and the transfer method can change the actual result. Use sharding and resumable transfers for a production upload rather than treating this SCP probe as a full-data transport implementation.

At the measured sustained local graph rate of about 155 peptides/s and about 19.9 kB of compressed graph inputs per peptide, production is roughly 3.1 MB/s. The short upload rate is above this average production rate, so graph preparation and shard upload could overlap. Under sustained comparable bandwidth, preparation need not incur a separate full 17-hour wait after all CPU work finishes. This overlap has not been implemented or measured as a complete pipeline.

To defer GPU rental until preparation finishes, upload shards to storage that can exist without a running GPU instance (for example, a separately retained data volume or object storage). Whether the current provider supports reattaching a volume or private-network/object-store reads remains unverified. Uploading directly to a rented, running GPU server still incurs that instance's rental time even when the GPU is idle.

An alternative is uploading only the approximately 2.33 GB filtered FASTA, preparing the graph cache on a cloud CPU instance, and handing the same retained storage to the GPU instance. That greatly reduces Mac-to-cloud upload volume, but depends on provider storage/transfer support and an independently measured CPU server rate; it is not automatically faster than this Mac. If all final embedding matrices must be used locally, approximately 140 GB still needs to be downloaded later; that direction's speed has not been measured. Cloud-side projection training can instead retain those matrices remotely and return small checkpoints/results or selected downstream features.

The random probe contained no user data. Both local and cloud probe files were deleted after testing. No data splits, encoding policy, training configuration, or model parameters changed. References: [local CPU benchmark](20261026-m5pro-cpu-preprocessing.md), [encoder GPU benchmark](20261026-encoder-extraction.md).

## Experimental compact graph storage

`scripts/benchmark_graph_storage.py` tested all 2,048 sampled peptides in 128 existing 16-peptide shards. It removes the repeated atom-feature prefix from bond features and reconstructs neighboring atom indices from existing indices; it packs binary feature columns into bits, retains all other feature columns as float32, and narrows indices only after checking exact recovery. Every restored component was checked for identical dtype, shape, and bytes. The original graph arrays and index files were not modified. This is an experimental storage codec; the production encoder reader has not been integrated with it.

Original graph NPZ files totaled 40,703,928 bytes; packed files totaled 13,497,195 bytes, a 3.02-fold reduction. Extrapolating packed bytes per sampled peptide gives approximately 79.9 decimal GB of graph data for 12,124,168 peptides, plus approximately 1.15 GB of token data and separate index metadata. At a sustained 4 MB/s, that is approximately 5.6 hours of upload. This is a sample extrapolation, not a full-data size or network guarantee. Encoding packed data from already loaded arrays took 1.42 seconds, and restoring it took 0.37 seconds across the sample on the Mac; those timings exclude original shard reading, filesystem writes, model execution, and parallel production integration.

Reproduce with the existing isolated CPU environment:

```sh
.venv/cpu-benchmark/bin/python scripts/benchmark_graph_storage.py \
  --input outputs/cpu-benchmark-20261026/sustained8/cpu-w8-r0 \
  --chunks 128 --output outputs/cpu-benchmark-20261026/compact-storage-fullsample.json
```

Retain the matching `index.json` to preserve sample IDs and order. Schema 1 is specific to this benchmark's eight GROVER graph components, not a generic molecular graph interchange format.
