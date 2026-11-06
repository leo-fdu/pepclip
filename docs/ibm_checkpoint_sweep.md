# Configuration-driven IBM checkpoint comparisons

Edit configs/ibm_checkpoint_sweep.json to select checkpoint paths/epochs, representations, tasks, explicit seeds, feature source, output directory and resources. JSON uses the standard library, with no new YAML dependency. Relative paths resolve against the repository root, irrespective of the shell working directory.

Preview only (no output files, inference, or evaluation):
```bash
PYTHONPATH=src .cache/ibm-venv/bin/python scripts/ibm_checkpoint_sweep.py --config configs/ibm_checkpoint_sweep.json --dry-run
```

When execution is explicitly desired, omit --dry-run. Use --prepare-only to generate projection caches without evaluation. The example describes epochs 5/10/15/20, two representations, four tasks and seeds 0–4: 160 evaluations. These are configuration values, not runner constants.

Configuration fields:
- checkpoints: list of objects with unique safe name, path, and expected epoch.
- representations: any unique subset of pepclip, grover-base, grover-pepclip-concat.
- tasks: any unique subset of antibacterial, antiviral, cpp, binding.
- seeds: explicit unique nonnegative uint32 integers.
- run, feature_source, ibm_root, grover_source, grover_checkpoint: paths.
- workers, threads_per_worker, hpo_trials: positive integers.

The benchmark protocol remains LightGBM, five-fold CV, patience 20, MCC classification / Spearman binding; this refactor does not change prediction models, splits, labels, normalization, target features, or scoring. Hyperparameter trial budget is configurable and must be kept matched for comparisons. Checkpoint queues run sequentially, with the configured worker count per queue. Raw GROVER does not depend on checkpoint; selecting it at multiple checkpoints repeats an identical representation evaluation, so the example omits it.

All original queue CLI commands remain supported. New --representations, --tasks and --seed-ids accept explicit lists and require --feature-run. They are stored in the queue configuration and used on resume. The sweep now requires --config rather than the former hard-coded --run interface.

The sweep stores a resolved configuration and checkpoint hashes in manifest.json; changing them requires a new output directory. Re-running an unchanged configuration resumes checksum-validated jobs; interrupted HPO starts again for that seed. A lock rejects concurrent sweep drivers. Each checkpoint's evaluation requests an Agent Caffeine hold if the CLI is installed; protection requires the app to be running.

Previously generated caches under outputs/ibm-checkpoint-sweep-5seeds-20261106 are retained unchanged. Their legacy manifest is intentionally not silently rewritten or adopted. The example uses a different output path; actual execution will build new projected caches from the same raw GROVER cache, without running the backbone.

Inspect status.json and each checkpoint queue's status.json for actual progress. No benchmark is run by configuration validation or dry-run. Downstream checkpoint comparison is exploratory; selecting by IBM test scores does not provide independent test validation.
