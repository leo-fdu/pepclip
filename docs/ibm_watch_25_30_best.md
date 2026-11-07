# Automatic PepCLIP IBM evaluation at epochs 25 and 30

`configs/ibm_watch_25_30_best.json` and `scripts/watch_ibm_checkpoints.py`
wrap the existing reprojection and queue commands without changing training,
projection or benchmark implementation. Only PepCLIP is evaluated: four tasks,
seeds 0–4, five-fold CV, up to 100 HPO trials and patience 20. Each distinct
checkpoint costs 20 runs (40–60 runs total).

```bash
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/watch_ibm_checkpoints.py \
  --config configs/ibm_watch_25_30_best.json --dry-run
# Remove --dry-run to watch and execute; unchanged configuration resumes state.
```

The watcher polls every 20 seconds. Epoch snapshots are atomically renamed by
the trainer only after last.pt and best.pt have been saved. Epoch 25 is copied
into the watcher's immutable checkpoint directory and queued immediately. The
presence of epoch 30 is the final-best barrier: final best is copied only then.
A digest of model arguments and every model tensor (including temperature),
not the serialized file bytes, detects whether best equals epoch 25 or 30.
Aliases refer to that evaluation instead of scheduling duplicates. A distinct
best receives an epoch metadata field in its isolated copy, derived from the
lowest validation loss in the original and continuation metrics. Original
checkpoint files are never modified. An inherited best from before epoch 21
is also supported, but is evaluated anew unless it matches epoch 25 or 30.

One benchmark command executes at a time. While training has not published
epoch 30, a new queue uses two workers with one compute thread each, alongside the
training configuration's ten compute threads and one reader. New queues after
epoch 30 use four workers and three threads. An already running queue retains
its original resources; epoch 30 and best may wait for epoch 25 to finish.
This increases overlap throughput but does not guarantee three idle CPU cores,
hard CPU affinity or absence of OOM.
Projection preparation uses one thread. Native compute thread environment
variables are also bounded per child command.

`outputs/ibm-watch-25-30-best-5seeds/status.json` records pending stages, active
child, completion and aliases. Per-stage logs and each queue's status.json
provide details. A lock prevents two watcher drivers; checkpoint hashes and
resolved configuration are verified on resume. Interrupted HPO jobs restart at
the seed boundary, as in the existing queue. Unexpected training exit before
epoch 30 stops the watcher and its active child. Restart the watcher after a
training restart only with an updated launch record pointing at that run.

Each active prepare/evaluate command requests Agent Caffeine protection if its
CLI is available. The app must actually be running; this does not itself ensure
sleep protection. No hold is added while simply polling an active training run.

Validation: four tests cover atomic-ready filenames, both best aliases,
distinct best metadata and source preservation, repeat discovery, and retaining
the virtual-environment executable path. Actual future epoch-25/30 benchmark
execution is not validated by these tests or dry-run.
