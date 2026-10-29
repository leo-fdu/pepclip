# IBM representation-transfer evaluation

The local deployment is explicitly requested by the user. Alignment training is
already complete; use `outputs/mps-85-15-b512-20epochs/best.pt`. The log's best
validation epoch is 18. Do not select a checkpoint using IBM test scores.

## Installation

Use Python 3.12 and an isolated environment (do not alter the training environment).
`requirements/ibm-benchmark-macos-lock.txt` records the complete installed local
environment; `ibm-benchmark.txt` pins the main dependencies:

```bash
python3.12 -m venv .cache/ibm-venv
.cache/ibm-venv/bin/python -m pip install -r requirements/ibm-benchmark.txt
git clone https://github.com/IBM/PeptideGeneralizationBenchmarks.git .cache/ibm-peptide-benchmarks
git -C .cache/ibm-peptide-benchmarks checkout 8779c803794ed962665cd0feefe65d000a138317
```

GROVER source and weights: follow `docs/full_embedding_extraction.md`. Source
commit is `40b6d97098e4508687912f3c05eca369fc2c6213`; weight SHA256 is
`47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de`.
The local source cache is `.cache/encoder-benchmark/grover-source` and weights
are `.cache/grover_base.pt`. These large/external artifacts are ignored by Git.

GROVER supports CPU/CUDA, not MPS. LightGBM uses CPU. If LightGBM cannot locate
OpenMP on macOS, set `DYLD_LIBRARY_PATH` to an existing compatible OpenMP library
directory, or install OpenMP in the target environment. Record the library used.
The SciPy 1.15.3 wheel failed to load on this macOS 27.0.1 machine
(`__thread_bss` zero-fill offset); 1.16.3 passed the import check. This is an
environment change, not a change to GROVER features or weights.
IBM's `run_all.sh` is not our entry point: it requests MPS for neural feature
extraction and has no PepCLIP registration.

## Feature export

```bash
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/export_ibm_features.py \
  --ibm-root .cache/ibm-peptide-benchmarks \
  --grover-source .cache/encoder-benchmark/grover-source \
  --grover-checkpoint .cache/grover_base.pt \
  --projection-checkpoint outputs/mps-85-15-b512-20epochs/best.pt \
  --output outputs/ibm-smoke-features --limit 16 --threads 4 --batch-size 8
```

For complete features remove `--limit` and change output to
`.cache/ibm-peptide-benchmarks/reps` (must not already exist). The output is
`grover-base_{dataset}.pickle` and `pepclip_{dataset}.pickle` for all eight
datasets. Each row remains in original CSV order, including duplicates and
protein-peptide pairs. Invalid SMILES abort; no 10–100 aa filter or sequence
reconstruction is applied. Noncanonical chemistry comes from official SMILES.

Weights, source hashes and pooling are checked against the alignment checkpoint.
CPU FP32 differs from training-cache CUDA FP16 autocast; both the training and
current metadata are saved. Both comparison arms share exactly the same raw
features. This is not claimed to reproduce CUDA cache bytes. On CUDA select
`--device cuda --precision cuda-fp16` explicitly. Projection stays FP32.

`export.json` records dataset and feature hashes, row counts, model load time,
per-batch graph/projection timings, atom counts, sequence lengths when present,
and completion status. Graph time includes graph construction and CPU transfer;
projection time includes output CPU transfer. First batch is cold. Smoke exports
are prohibited from writing to the official `reps` directory.

## Smoke evaluation (not a benchmark result)

```bash
.cache/ibm-venv/bin/python scripts/smoke_ibm_evaluation.py \
  --ibm-root .cache/ibm-peptide-benchmarks \
  --features outputs/ibm-smoke-features \
  --output outputs/ibm-smoke-evaluation.json --hpo-space ibm-yaml
```

This imports the real IBM evaluator, loads official c-cpp partitions and runs
one LightGBM trial, two-fold CV on the small c-cpp prefix, for each feature arm.
It performs an untimed one-tree LightGBM warmup before measuring HPO.
Its temporary stratified split is ONLY for integration verification. It reports
training/prediction time, not publishable performance or official runtime.

## Full evaluation

After a complete feature export, run from the PepCLIP checkout:

```bash
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/run_ibm_evaluation.py \
  --ibm-root .cache/ibm-peptide-benchmarks --mode interpolation \
  --dataset c-cpp --representation grover-base --hpo-space ibm-yaml
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/run_ibm_evaluation.py \
  --ibm-root .cache/ibm-peptide-benchmarks --mode interpolation \
  --dataset c-cpp --representation pepclip --hpo-space ibm-yaml
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/run_ibm_evaluation.py \
  --ibm-root .cache/ibm-peptide-benchmarks --mode ood \
  --dataset cpp --representation pepclip --hpo-space ibm-yaml
```

The runner checks full feature counts/checksums and unchanged source CSVs,
refuses existing result/run files, and changes directory for the official
interpolation evaluator's relative `reps/` path.

**HPO compatibility:** AutoPeptideML 2.1.0 (also inspected 2.0.7) expects flat
per-model overrides, whereas IBM passes a complete nested search space. Direct
execution silently uses package defaults (e.g. 8–256 trees instead of IBM's
10–500). `--hpo-space ibm-yaml` adapts the trainer to use exactly IBM's YAML
search variables, with `n_jobs` explicitly passed as an execution setting.
`--hpo-space upstream-default` preserves the current unmodified combination's
behavior. The choice is mandatory, saved in each `.run.json`, and must be the
same for both arms. These policies are not interchangeable results; neither
establishes reproduction of the paper's historical dependency environment.
The smoke test asserts that the effective search space matches the IBM YAML.

Repeat for other tasks and canonical/noncanonical interpolation datasets.
Keep budgets identical. At the pinned commit interpolation defaults to 5 seeds;
OOD defaults to 25 seeds. Both call HPO with 100 trials, 5 folds, 10 jobs. The
interpolation CLI's `n_trials` argument is unused internally: changing it does
not reduce the HPO budget. Official HPO may early-stop. Do not silently call a
reduced run a complete benchmark. Official scripts can overwrite result CSVs;
use a fresh experiment checkout/output location for independent runs.

Binding additionally requires `reps/binding-c-targets.pickle` and
`reps/binding-nc-targets.pickle`. Generate them with the official
`protein_data_binding(device='cpu')` function in `represent_peptides.py`
(ESM2-8M; it may download weights) before binding evaluation:

```bash
# From the IBM checkout, after full feature export has created reps/:
../ibm-venv/bin/python rep_transfer/represent_peptides.py binding-targets unused --device cpu
```
Use the same target
features for both arms. These files are separate from our peptide encoder.

Before scientific reporting, audit duplicate/near-duplicate molecules across
official splits and overlap with the existing UniRef90 FASTA. Keep official
results separate from any newly decontaminated splits. Noncanonical structure
acceptance does not demonstrate generalization; downstream results are needed.

## Local validation and speed (2026-10-29)

On Apple M5 Pro / 24 GiB / macOS 27.0.1, CPU 4 threads, batch 8:
128 real IBM records (first 16 from each of eight datasets) took 10.747 seconds
in GROVER, including graph construction: 11.91 records/s. Projection took
0.054 seconds in total. These prefixes are not a representative throughput
sample; do not infer a precise full-run duration. The first batch is included
and upstream RDKit deprecation logging was enabled.

A separate warm projection-only test on 512 real UniRef90 GROVER vectors, five
repeats, measured 24,601 vectors/s; this excludes graph encoding and file I/O.
Repeated encoding of two canonical/noncanonical examples was bitwise identical;
the encoder was frozen/in eval mode and projected vectors had unit norm.
All 21,037 official input records parsed with RDKit. Duplicate molecules were
counted, not removed; this is not a cross-split leakage audit.

Detailed measurements: `docs/benchmarks/20261029-ibm-local-smoke.json`.
A second run on the first 256 c-cpp records took 23.259 seconds for GROVER
(11.01 records/s) and 0.100 seconds for projection. Its downstream integration
test, using IBM YAML with one trial/two folds/one job after warmup, took 0.271
seconds for raw GROVER and 0.377 seconds for PepCLIP. This reduced budget is
not an estimate of the official full HPO time or an accuracy comparison.
Full benchmark scores, binding target feature generation, full-data timings,
and training-overlap/similarity audits remain unexecuted.
