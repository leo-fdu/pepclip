# Cloud CPU preprocessing benchmark, 2026-10-26

The existing RTX 4090 instance exposes 10 vCPUs, approximately 31 GiB RAM, and an AMD EPYC 7402 CPU model string. Guest topology reports one socket, 10 cores, one thread per core, with CPU IDs 0–9 mapped to distinct guest core IDs. This is virtual topology and does not establish physical-core dedication, host SMT settings, or host contention. No additional hardware threads are visible to the guest. Increasing process count is software oversubscription, not enabling hardware SMT.

The same CPU-only graph pipeline and seed-42 reservoir sample used on the Mac were tested without encoder loading or GPU execution. It includes sequence-to-SMILES, unchanged upstream MolGraph/BatchMolGraph, float32/int64 compressed NPZ writes/fsync, and exact decompression checks. Each spawned worker limits native tensor/BLAS threads to one. Sample/source hashes and raw timings are in [the machine-readable results](20261026-epyc7402-cpu.json).

| Processes | Peptides per timed repeat | Repeats | Median peptides/s |
|---|---:|---:|---:|
| 1 | 512 | 2 | 6.84 |
| 4 | 512 | 2 | 21.50 |
| 8 | 512 | 2 | 33.22 |
| 10 | 512 | 2 | 36.88 |
| 10 | 2,048 | 3 | 41.99 |
| 16 | 2,048 | 3 | 42.41 |
| 20 | 2,048 | 3 | 42.22 |

Oversubscription tests used the same 2,048 peptides and three timed repeats per process count. Sixteen processes measured 42.52, 42.41, and 42.32 peptides/s; twenty measured 41.99, 42.22, and 42.36. Median improvements over ten processes are approximately 1.0% and 0.6%, respectively. These small differences do not establish a useful throughput improvement, and extra processes do not reveal or enable additional hardware threads. Keep ten workers for the present instance. All process counts reproduced the same first-chunk array checksum as the Mac benchmark, and every saved chunk passed exact restoration checks.

The larger 10-process test measured 42.36, 41.74, and 41.99 peptides/s. Extrapolating the median gives 66.2 hours for ten million or 80.2 hours for all 12,124,168 peptides. Single-process ESM tokenization plus compressed cache writes and exact restoration measured 1,546.94 peptides/s, equivalent to 2.18 hours for the full dataset. Sequential graph and token preparation therefore extrapolates to 82.4 hours. Worker/model startup, full-FASTA parsing/index management, long-run load changes, production resumption, and network work are not represented by this estimate. This is a roughly 2.5-minute larger-window measurement, not a multi-day run.

Peak reported worker RSS in the 10-process larger-window test was 520.3 MiB per worker. This is a per-worker maximum, not measured total peak memory of a production CPU/GPU pipeline. The test does not indicate a need for 108 GB RAM to run this particular preprocessing workload. The Mac's previously measured sustained rate was approximately 155 peptides/s, substantially above this VM; core count alone does not predict throughput across CPU models and instance allocations.

## Interpreting the proposed dual-4090 / 22-vCPU instance

If a 22-vCPU instance delivered the same effective throughput per allocated CPU as this VM at 10 processes, linear extrapolation would give approximately 92.4 peptides/s and 36.5 hours of graph preparation. This is a planning scenario, not a measured result or a confidence bound. A different CPU model, SMT allocation, CPU quota, contention, or scaling efficiency can change it substantially. The CPU model and price of the proposed instance are unknown.

Prepared-input GPU measurements were approximately 156.5 peptides/s for GROVER and 586.5 peptides/s for ESM2 on one 4090. Sequential single-GPU extraction totals approximately 27.3 hours for the complete dataset. Perfectly balanced two-GPU sharding would have a GPU-stage lower estimate of 13.6 hours; multi-GPU execution and scaling have not been implemented or measured. Assigning one complete encoder to each GPU instead would leave the approximately 21.5-hour GROVER stage as the longer branch.

At a hypothetical 92.4 CPU peptides/s, CPU preparation could feed less than one 4090's measured GROVER capacity. Two GPUs would therefore not be continuously saturated by this CPU producer. With separate stages, graph preparation plus ideally balanced dual-GPU work is approximately 50.1 hours before tokenization, storage, startup, and orchestration overhead. An asynchronous producer/consumer pipeline could overlap those stages and approach the approximately 36.5-hour CPU bottleneck, with GPU work completing after the final shards; this pipeline is not implemented or validated. Running CPU and GPU work simultaneously can also change both measured rates. Renting two GPUs for the larger CPU allocation may still be practical, but the total price cannot be compared without hourly costs.

Cloud preprocessing would require uploading only the approximately 2.33 GB filtered FASTA instead of the full graph cache. Storage still matters: final dual-side FP32 embeddings alone are approximately 140 GB; retained experimental compact graph inputs add approximately 80 GB. The existing instance's roughly 97 GB root disk cannot retain both. RAM allocation is not a replacement for persistent disk capacity.

Reproduce from the previously validated cloud environment:

```sh
export PYTHONPATH=src
python scripts/benchmark_cpu_graphs.py --grover-source /path/to/grover-source \
  --sample /path/to/sample.json --output outputs/cpu-cloud-scaling/results.json \
  --workers 1 4 8 10 --samples 512 --repeats 2
python scripts/benchmark_cpu_graphs.py --grover-source /path/to/grover-source \
  --sample /path/to/sample.json --output outputs/cpu-cloud-sustained/results.json \
  --workers 10 --samples 2048 --repeats 3
python scripts/benchmark_cpu_tokens.py --sample /path/to/sample.json \
  --output outputs/cpu-cloud-tokens/results.json --repeats 3
python scripts/benchmark_cpu_graphs.py --grover-source /path/to/grover-source \
  --sample /path/to/sample.json --output outputs/cpu-cloud-oversubscribe/results.json \
  --workers 16 20 --samples 2048 --repeats 3
```

No full dataset job was started, no dataset split or model configuration changed, and no new dependencies were installed for this cloud CPU test. All CPU benchmark jobs completed; GPU utilization was 0% with 1 MiB occupied at the final check. SSH sessions were closed.
