# IBM checkpoint sweep (2026-11-06)

Compare epochs 5, 10, 15, and 20, each as PepCLIP and raw-GROVER + PepCLIP concatenation. Four tasks and seeds 0–4 give 160 evaluations. Keep the existing IBM natural-to-modified protocol: LightGBM, five CV folds, at most 100 Optuna trials, patience 20, MCC classification / Spearman binding. Target representations are appended once. No raw-GROVER evaluations are repeated.

Use the existing IBM environment:
```bash
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/ibm_checkpoint_sweep.py \
  --run outputs/ibm-checkpoint-sweep-5seeds-20261106
```

The same command resumes completed feature preparation and evaluation jobs after checksum checks; interrupted seed HPO restarts. Do not run two sweep drivers for the same output. The driver holds a sweep lock to reject concurrent drivers. Preparation uses existing raw GROVER and binding target caches without invoking either backbone, validates source hashes, checkpoint epochs and encoder provenance, and writes separate projected caches. All four feature caches can be prepared first using --prepare-only.

The four evaluation queues run sequentially, each with five workers and three compute threads. Each queue requests its own Agent Caffeine command hold, so the full sweep does not depend on one hold exceeding the 24-hour limit. The app must be running for sleep protection. Check agent-caffeine status --json; no protection is claimed if the app is unavailable.

Inspect the sweep status.json and each epoch-NN/status.json for actual progress. Each queue retains predictions, CV indices, HPO trials and per-seed outputs. Complete means all requested evaluations succeeded, not just cache preparation. A rough initial estimate from the previous 40 jobs is 33 hours at five workers; HPO duration and machine contention can change it.

The existing best.pt result corresponds to validation-loss selection; this sweep is exploratory downstream comparison and does not turn the IBM test set into an independent checkpoint-selection validation set.
