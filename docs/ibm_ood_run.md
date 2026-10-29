# PepCLIP natural-to-modified run

The user selected only PepCLIP, four tasks, 25 seeds per task, CPU capacity
15 threads. The queue uses three processes, each with five compute threads;
it does not promise constant 100% utilization during I/O or the last jobs.

## Explicit protocol

- Official IBM data at `8779c803794ed962665cd0feefe65d000a138317`.
- All canonical rows train; all modified rows test. No domain-interpolation
  partitions, no new length filter, and no use of test labels for selection.
- `best.pt` from the completed alignment run; CPU FP32 GROVER extraction,
  the trained B head, L2 normalization. No new alignment training.
- LightGBM with the IBM YAML search space, at most 100 trials, five-fold CV,
  patience 20. The unused CLI 200-trial default is not used.
- Classification maximizes MCC; binding maximizes Spearman (`spcc`). This
  corrects the current dependency's implicit maximize-MSE regression default.
- Explicit TPE seed, CV seed and LightGBM seed equal to run seed 0–24.
  Explicit sampling seeds and the objective correction are protocol changes
  from blindly executing current upstream defaults, not a claim of exact
  reproduction of the historical leaderboard runtime.
- Binding targets use official AutoPeptideML `RepEngineLM('esm2-8m')`, same
  residue pooling and CPU autocast policy, explicit eval/frozen parameters.
  Batch size is four instead of 64 to bound memory with concurrent jobs.
  The official 1022-residue truncation is retained and truncated row indices
  are recorded (two canonical binding target rows exceed the limit).
- Metrics are retained for each of the five best CV models, as in the official
  OOD evaluator. `run_seed` and `fold` disambiguate the upstream overlapping
  `seed = run_seed + fold` field. Per-sample predictions are retained.

## Start and resume

After installing the environment and caching ESM2-8M as described in
`ibm_benchmark.md`, start from the project root:

```bash
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/ibm_ood_queue.py \
  --run outputs/ibm-pepclip-ood-20261029 \
  --ibm-root .cache/ibm-peptide-benchmarks \
  --grover-source .cache/encoder-benchmark/grover-source \
  --grover-checkpoint .cache/grover_base.pt \
  --projection-checkpoint outputs/mps-85-15-b512-20epochs/best.pt \
  --workers 3 --threads 5 --seeds 25 --trials 100
```

The 109 jobs consist of eight peptide feature jobs, one binding-target job,
and 100 evaluation jobs. Preparation has priority; evaluation starts as slots
become available and its prerequisites complete. Workers run offline after
model download. Each worker's BLAS/OpenMP threads are capped at five.

Resume only when the old coordinator has stopped:

```bash
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/ibm_ood_queue.py \
  --run outputs/ibm-pepclip-ood-20261029 --resume
```

An exclusive lock prevents duplicate coordinators. Input, model, source and
dependency fingerprints must match on resume. Completed jobs are checksum
verified and skipped. An interrupted job restarts from scratch in a new
`attempt-NNN` directory; previous attempts are retained. This is job-level
recovery, not mid-trial or mid-feature-batch continuation. A worker failure
stops the queue and records the error; there is no silent automatic retry.

`status.json` contains coordinator/worker PIDs and completion counts.
`jobs/*/attempt-*/worker.log` contains per-job progress; evaluation jobs also
write `progress.json` after each trial. `done.json` is written only after
successful output validation. The four task CSVs aggregate completed seeds;
they are partial until status reports `complete` and all 100 evaluations finish.

Sleep pauses work unless the machine is kept awake. Agent Caffeine must be
running for its hold to take effect; this launcher does not change power settings.

## Validation

Queue counts, binding dependencies, completion markers and corrupt-output
rejection have standard-library tests in `tests/test_ibm_queue.py`. Synthetic
classification and regression workers completed five-fold, one-trial checks
using the real installed trainer; objectives and prediction shapes were checked.
These tests verify the runner, not benchmark accuracy. Training-data overlap
and cross-domain similarity audits remain separate work before scientific claims.
