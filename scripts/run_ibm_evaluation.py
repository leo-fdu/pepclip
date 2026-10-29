"""Run pinned IBM evaluator with an explicit choice of HPO compatibility policy."""
import argparse
import copy
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path


def trainer_class(policy):
    if version("autopeptideml") != "2.1.0":
        raise ValueError("This compatibility bridge requires autopeptideml==2.1.0")
    from autopeptideml.train import OptunaTrainer
    if policy == "upstream-default":
        return OptunaTrainer
    if policy != "ibm-yaml":
        raise ValueError(policy)

    class IBMTrainer(OptunaTrainer):
        def _get_hpspace(self, models, custom_hpspace):
            # APML 2.1 expects flat per-model overrides. IBM supplies a complete
            # nested space; passing it unchanged otherwise silently ignores it.
            space = copy.deepcopy(custom_hpspace)
            if set(space["models"]["elements"]) != set(models):
                raise ValueError("IBM search-space/model mismatch")
            for variables in space["models"]["elements"].values():
                # Execution parallelism, not a searched hyperparameter.
                variables["n_jobs"] = {"type": "fixed", "value": self.n_jobs}
            return space

    return IBMTrainer


def load_evaluator(root, mode, policy):
    filename = "evaluation.py" if mode == "interpolation" else "evaluation_ood.py"
    spec = importlib.util.spec_from_file_location("ibm_official", root / "rep_transfer" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OptunaTrainer = trainer_class(policy)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibm-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("interpolation", "ood"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--representation", choices=("grover-base", "pepclip"), required=True)
    parser.add_argument("--hpo-space", choices=("ibm-yaml", "upstream-default"), required=True)
    args = parser.parse_args()
    root = args.ibm_root.resolve()
    tasks = ("binding", "cpp", "antibacterial", "antiviral")
    valid = tasks if args.mode == "ood" else tuple(f"{k}-{t}" for k in ("c", "nc") for t in tasks)
    if args.dataset not in valid:
        parser.error(f"Dataset must be one of {valid}")
    report = json.loads((root / "reps/export.json").read_text())
    if report["status"] != "complete" or report["smoke"]:
        parser.error("Complete, non-smoke feature export required")
    names = [args.dataset] if args.mode == "interpolation" else [f"{k}-{args.dataset}" for k in ("c", "nc")]
    from pepclip.utils import file_sha256
    for name in names:
        entry = report["datasets"][name]
        path = root / "reps" / f"{args.representation}_{name}.pickle"
        if entry["source_rows"] != entry["exported_rows"] or file_sha256(path) != entry["files"][args.representation]["sha256"]:
            parser.error(f"Incomplete or changed features: {name}")
        if file_sha256(root / "downstream_data" / f"{name}.csv") != entry["source_sha256"]:
            parser.error(f"Source rows changed: {name}")
        if name.endswith("binding") and not (root / "reps" / f"binding-{name.split('-')[0]}-targets.pickle").exists():
            parser.error("Generate official binding target representations first")
    result = (root / "Results/no-generalisation" / f"{args.dataset}_lightgbm_pre_0.0_post_0.0_{args.representation}.csv"
              if args.mode == "interpolation" else root / "Results/standard" / f"{args.dataset}_lightgbm_{args.representation}.csv")
    metadata = result.with_suffix(".run.json")
    if result.exists() or metadata.exists():
        parser.error("Result or run record exists; use a fresh experiment checkout")
    module = load_evaluator(root, args.mode, args.hpo_space)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(status="running", hpo_space=args.hpo_space, benchmark_commit=report["benchmark_commit"],
                    n_trials=100, n_folds=5, n_jobs=10,
                    n_seeds=5 if args.mode == "interpolation" else 25)
    metadata.write_text(json.dumps(settings, indent=2) + "\n")
    os.chdir(root)
    module.main(args.dataset, "lightgbm", args.representation)
    settings["status"] = "complete"
    metadata.write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
