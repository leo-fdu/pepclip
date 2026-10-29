"""Check IBM imports, official partitions and one tiny HPO; NOT benchmark scores."""
import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibm-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hpo-space", choices=("ibm-yaml", "upstream-default"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    import numpy as np
    import pandas as pd
    from hestia import HestiaGenerator
    from sklearn.model_selection import train_test_split
    from autopeptideml.train import evaluate
    from run_ibm_evaluation import trainer_class
    OptunaTrainer = trainer_class(args.hpo_space)
    from lightgbm import LGBMClassifier

    spec = importlib.util.spec_from_file_location("ibm_evaluation", args.ibm_root / "rep_transfer/evaluation.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    df = pd.read_csv(args.ibm_root / "downstream_data/c-cpp.csv")
    hdg = HestiaGenerator(df)
    hdg.from_precalculated(str(args.ibm_root / "partitions/c-cpp.gz"))
    thresholds = [str(th) for th, _ in hdg.get_partitions(filter=0.185)]
    if not thresholds:
        raise ValueError("Official partitions did not load")
    export = json.loads((args.features / "export.json").read_text())
    if export["status"] != "complete":
        raise ValueError("Incomplete feature export")
    result = dict(kind="smoke-only-not-benchmark", official_partition_thresholds=thresholds,
                  timing="after untimed one-tree LightGBM warmup per representation",
                  n_trials=1, n_folds=2, n_jobs=1, models={})
    result["hpo_space"] = args.hpo_space
    for name in ("grover-base", "pepclip"):
        with (args.features / f"{name}_c-cpp.pickle").open("rb") as handle:
            x = np.asarray(pickle.load(handle))
        y = df.labels.to_numpy()[:len(x)]
        train, test = train_test_split(np.arange(len(x)), test_size=0.25, stratify=y, random_state=42)
        space, _ = official.define_hpspace("lightgbm", "class")
        LGBMClassifier(n_estimators=1, n_jobs=1, verbosity=-1).fit(x[train], y[train])
        trainer = OptunaTrainer(task="class")
        started = time.monotonic()
        trainer.hpo(x={"default": x[train]}, y=y[train], models=["lightgbm"],
                    n_folds=2, n_trials=1, patience=1, n_jobs=1, random_state=42,
                    verbose=0, custom_hpspace=space)
        predictions = trainer.best_model.predict_proba({"default": x[test]})[0][:, 1]
        if args.hpo_space == "ibm-yaml":
            effective = trainer.hpspace["models"]["elements"]["lightgbm"]
            expected = dict(space["models"]["elements"]["lightgbm"],
                            n_jobs={"type": "fixed", "value": 1})
            if effective != expected:
                raise AssertionError("Effective HPO space differs from IBM YAML")
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite downstream predictions")
        metrics = evaluate(predictions, y[test], pred_task="class")
        result["models"][name] = dict(shape=list(x.shape), predictions=predictions.tolist(),
                                      train_and_predict_seconds=time.monotonic() - started,
                                      metrics={k: float(v) for k, v in metrics.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
