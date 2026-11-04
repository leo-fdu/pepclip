"""Resumable process queue: prepare features, then 4 tasks x 25 independent seeds.

Completed jobs are reused only after checksum verification. Interrupted jobs
restart in a new attempt directory; trials within a seed are not resumed.
"""
import argparse
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

TASKS = ("antibacterial", "antiviral", "cpp", "binding")
ROOT = Path(__file__).resolve().parents[1]


def components(representation):
    return ("grover-base", "pepclip") if representation == "grover-pepclip-concat" else (representation,)


def load_features(folder, dataset, representation):
    import numpy as np
    import pickle
    arrays = []
    for name in components(representation):
        with (folder / f"{name}_{dataset}.pickle").open("rb") as f:
            arrays.append(np.asarray(pickle.load(f), dtype=np.float32))
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid feature matrix")
    return np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]


def evaluation_jobs(seeds, representations):
    return {f"eval-{task}-{seed:02d}:{rep}": []
            for seed in range(seeds) for task in TASKS for rep in representations}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def job_specs(seeds):
    jobs = {f"feature-{k}-{t}": [] for t in TASKS for k in ("c", "nc")}
    jobs["targets"] = []
    for seed in range(seeds):
        for task in TASKS:
            jobs[f"eval-{task}-{seed:02d}"] = [f"feature-c-{task}", f"feature-nc-{task}"]
            if task == "binding":
                jobs[f"eval-{task}-{seed:02d}"].append("targets")
    return jobs


def completed(run, job):
    path = run / "jobs" / job / "done.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for filename, digest in data["files"].items():
        if sha(run / filename) != digest:
            raise ValueError(f"Changed completed output: {filename}")
    return data


def mark_done(run, job, attempt, files):
    write_json(run / "jobs" / job / "done.json", dict(
        attempt=str(attempt.relative_to(run)),
        files={str(p.relative_to(run)): sha(p) for p in files}, finished=time.time()))


def cached_inputs(source, ibm_root, representation):
    """Validate complete preparation jobs and bind them to current CSV bytes."""
    files = []
    for job in job_specs(0):
        data = completed(source, job)
        if data is None:
            raise ValueError(f"Missing cached preparation job: {job}")
        files.append(source / "jobs" / job / "done.json")
        files.extend(source / name for name in data["files"])
        if job.startswith("feature-"):
            dataset = job.removeprefix("feature-")
            folder = source / data["attempt"] / "reps"
            report = json.loads((folder / "export.json").read_text())
            entry = report["datasets"][dataset]
            if (report["status"] != "complete" or report["smoke"]
                    or entry["source_rows"] != entry["exported_rows"]
                    or sha(ibm_root / "downstream_data" / f"{dataset}.csv") != entry["source_sha256"]
                    or any(sha(folder / f"{name}_{dataset}.pickle") != entry["files"][name]["sha256"]
                           for name in components(representation))):
                raise ValueError(f"Incomplete or changed cached dataset: {dataset}")
    return files


def feature_job(run, job, attempt, config):
    import export_ibm_features
    dataset = job.removeprefix("feature-")
    sys.argv = ["export_ibm_features", "--ibm-root", config["ibm_root"],
        "--grover-source", config["grover_source"], "--grover-checkpoint", config["grover_checkpoint"],
        "--projection-checkpoint", config["projection_checkpoint"], "--output", str(attempt / "reps"),
        "--datasets", dataset, "--threads", str(config["threads"]), "--batch-size", "8",
        "--quiet-rdkit-warnings"]
    export_ibm_features.main()
    mark_done(run, job, attempt, list((attempt / "reps").iterdir()))


def target_job(run, job, attempt, config):
    import numpy as np
    import pandas as pd
    import torch
    from autopeptideml.reps.lms import RepEngineLM
    torch.set_num_threads(config["threads"])
    encoder = RepEngineLM("esm2-8m", average_pooling=True)
    if encoder.model.config._commit_hash != config["target_model_revision"]:
        raise ValueError("Target encoder revision changed")
    encoder.move_to_device("cpu")
    encoder.model.eval().requires_grad_(False)
    meta = dict(model="facebook/esm2_t6_8M_UR50D", revision=encoder.model.config._commit_hash,
        pooling="official RepEngineLM residue mean", truncation=encoder.max_len(),
        fp16_flag=encoder.fp16, batch_size=4, datasets={})
    for kind in ("c", "nc"):
        df = pd.read_csv(Path(config["ibm_root"]) / "downstream_data" / f"{kind}-binding.csv")
        sequences = df.seq1.tolist()
        parts = []
        for i in range(0, len(sequences), 4):
            parts.append(encoder.compute_reps(sequences[i:i+4], batch_size=4))
            print(json.dumps(dict(targets=kind, rows=min(i+4, len(sequences)), total=len(sequences))), flush=True)
        values = np.concatenate(parts).astype(np.float32)
        if values.shape != (len(df), 320) or not np.isfinite(values).all():
            raise ValueError("Invalid target features")
        np.save(attempt / f"{kind}-targets.npy", values)
        meta["datasets"][kind] = dict(rows=len(df), truncated_rows=[i for i,s in enumerate(sequences) if len(s)>encoder.max_len()])
    write_json(attempt / "targets.json", meta)
    mark_done(run, job, attempt, [attempt / "c-targets.npy", attempt / "nc-targets.npy", attempt / "targets.json"])


def train_job(run, job, attempt, config):
    import numpy as np
    import pandas as pd
    import optuna
    from autopeptideml.train import evaluate
    from run_ibm_evaluation import load_evaluator, trainer_class
    base_job, _, job_representation = job.partition(":")
    _, task, seed_text = base_job.split("-")
    seed = int(seed_text)
    source = Path(config.get("feature_run", run))
    representation = job_representation or config.get("representation", "pepclip")
    np.random.seed(seed)
    x, y = {}, {}
    for kind in ("c", "nc"):
        data = completed(source, f"feature-{kind}-{task}")
        folder = source / data["attempt"] / "reps"
        report = json.loads((folder / "export.json").read_text())
        entry = report["datasets"][f"{kind}-{task}"]
        if report["status"] != "complete" or report["smoke"] or entry["source_rows"] != entry["exported_rows"]:
            raise ValueError("Incomplete benchmark features")
        x[kind] = load_features(folder, f"{kind}-{task}", representation)
        df = pd.read_csv(Path(config["ibm_root"]) / "downstream_data" / f"{kind}-{task}.csv")
        y[kind] = df.labels.to_numpy()
        if task == "binding":
            target = completed(source, "targets")
            x[kind] = np.concatenate([x[kind], np.load(source / target["attempt"] / f"{kind}-targets.npy")], axis=1)
        if len(x[kind]) != len(y[kind]) or not np.isfinite(x[kind]).all():
            raise ValueError("Feature/label mismatch")
    pred_task = "reg" if task == "binding" else "class"
    metric = "spcc" if task == "binding" else "mcc"
    official = load_evaluator(Path(config["ibm_root"]), "ood", "ibm-yaml")
    space, _ = official.define_hpspace("lightgbm", pred_task)
    # Explicit seed in LightGBM as well as CV and Optuna; record this protocol
    # correction rather than claiming identical historical default behavior.
    space["models"]["elements"]["lightgbm"]["random_state"] = {"type": "fixed", "value": seed}
    trainer = trainer_class("ibm-yaml")(task=pred_task, direction="maximize", metric=metric)
    original = optuna.create_study
    def create_study(**kwargs):
        kwargs["sampler"] = optuna.samplers.TPESampler(seed=seed)
        study = original(**kwargs)
        def progress(study, trial):
            write_json(attempt / "progress.json", dict(task=task, run_seed=seed,
                trials=len(study.trials), best_value=study.best_value, updated=time.time()))
        original_optimize = study.optimize
        def optimize(*args, **kwargs):
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), progress]
            return original_optimize(*args, **kwargs)
        study.optimize = optimize
        return study
    optuna.create_study = create_study
    started = time.monotonic()
    try:
        trainer.hpo(x={"default": x["c"]}, y=y["c"], models=["lightgbm"],
            n_folds=5, n_trials=config["trials"], patience=20, n_jobs=config["threads"],
            random_state=seed, verbose=0, custom_hpspace=space)
    finally:
        optuna.create_study = original
    metrics, predictions = [], []
    for fold, model in enumerate(trainer.best_model.models):
        pred = model.predict(x["nc"]) if pred_task == "reg" else model.predict_proba(x["nc"])[:, 1]
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite predictions")
        predictions.append(pred)
        metrics.append(dict(evaluate(pred, y["nc"], pred_task), run_seed=seed, fold=fold, seed=seed+fold))
    pd.DataFrame(metrics).to_csv(attempt / "metrics.csv", index=False)
    np.savez_compressed(attempt / "predictions.npz", predictions=np.stack(predictions), labels=y["nc"])
    np.savez_compressed(attempt / "cv_indices.npz", **{f"{kind}_{i}": a for i,pair in enumerate(trainer.train_folds) for kind,a in zip(("train","valid"),pair)})
    trainer.study.trials_dataframe().to_csv(attempt / "trials.csv", index=False)
    write_json(attempt / "result.json", dict(task=task, representation=representation, run_seed=seed, objective=metric,
        direction="maximize", trials=len(trainer.study.trials), best_config=trainer.best_config,
        effective_space=trainer.hpspace, seconds=time.monotonic()-started,
        train_rows=len(y["c"]), test_rows=len(y["nc"])))
    mark_done(run, job, attempt, [attempt / f for f in ("metrics.csv", "predictions.npz", "cv_indices.npz", "trials.csv", "result.json")])


def aggregate(run, done, representation="pepclip"):
    for task in TASKS:
        rows = []
        for job, data in done.items():
            if job.startswith(f"eval-{task}-") and (":" not in job or job.endswith(":" + representation)):
                with (run / data["attempt"] / "metrics.csv").open() as f:
                    rows.extend(csv.DictReader(f))
        if rows:
            path = run / f"{task}_{representation}.csv"
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(path)


def reuse_evaluations(source, run, config):
    """Copy verified baseline evaluations without changing the original run."""
    previous = json.loads((source / "config.json").read_text())
    if previous["representation"] != "grover-base":
        raise ValueError("Reuse source must be raw GROVER")
    for key in ("feature_run", "ibm_root", "threads", "trials", "mode", "device", "precision",
                "patience", "cv_folds", "objectives", "sampler", "model_seed",
                "package_versions", "benchmark_commit", "protocol", "target_model_revision"):
        if previous.get(key) != config.get(key):
            raise ValueError(f"Reuse protocol mismatch: {key}")
    # The queue itself has changed only to support concatenation and scheduling.
    # Every other recorded input must retain the exact hash of the baseline run.
    for name, digest in previous["input_hashes"].items():
        if Path(name).resolve() == Path(__file__).resolve():
            continue
        if config["input_hashes"].get(name) != digest or sha(name) != digest:
            raise ValueError(f"Reuse input mismatch: {name}")
    imported = []
    for seed in range(min(previous["seeds"], config["seeds"])):
        for task in TASKS:
            old_job = f"eval-{task}-{seed:02d}"
            data = completed(source, old_job)
            if data is None:
                continue
            old_attempt = source / data["attempt"]
            result = json.loads((old_attempt / "result.json").read_text())
            if result["representation"] != "grover-base" or result["run_seed"] != seed or result["task"] != task:
                raise ValueError("Reuse result identity mismatch")
            job = old_job + ":grover-base"
            attempt = run / "jobs" / job / "attempt-001"
            shutil.copytree(old_attempt, attempt)
            files = [attempt / Path(name).name for name in data["files"]]
            mark_done(run, job, attempt, files)
            imported.append(dict(job=job, source=str(old_attempt),
                                 source_done_sha256=sha(source / "jobs" / old_job / "done.json")))
    config["reused_evaluations"] = dict(source=str(source),
        source_config_sha256=sha(source / "config.json"), jobs=imported)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ibm-root", type=Path)
    p.add_argument("--grover-source", type=Path)
    p.add_argument("--grover-checkpoint", type=Path)
    p.add_argument("--projection-checkpoint", type=Path)
    p.add_argument("--representation", choices=("pepclip", "grover-base", "grover-pepclip-concat", "grover-and-concat"), default="pepclip")
    p.add_argument("--reuse-grover-run", type=Path, help="Import completed GROVER seeds into a combined run")
    p.add_argument("--feature-run", type=Path, help="Reuse validated preparation jobs from an existing run")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--seeds", type=int, default=25)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker")
    p.add_argument("--attempt", type=Path)
    args = p.parse_args()
    run = args.run.resolve()
    if args.worker:
        config = json.loads((run / "config.json").read_text())
        if args.worker.startswith("feature-"):
            feature_job(run, args.worker, args.attempt, config)
        elif args.worker == "targets":
            target_job(run, args.worker, args.attempt, config)
        else:
            train_job(run, args.worker, args.attempt, config)
        return
    if min(args.workers, args.threads, args.seeds, args.trials) < 1:
        p.error("Worker, thread, seed and trial counts must be positive")
    if not args.resume and args.representation == "grover-and-concat" and not args.feature_run:
        p.error("Combined evaluation requires --feature-run")
    if args.reuse_grover_run and (args.resume or args.representation != "grover-and-concat"):
        p.error("--reuse-grover-run requires a new combined run")
    if run.exists() and not args.resume:
        p.error("Run exists; use --resume")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        config = json.loads((run / "config.json").read_text())
    else:
        paths = {key: str(getattr(args, key).resolve()) for key in ("ibm_root", "grover_source", "grover_checkpoint", "projection_checkpoint")}
        config = dict(paths, workers=args.workers, threads=args.threads, seeds=args.seeds, trials=args.trials,
            mode="natural-to-modified", representation=args.representation, device="cpu", precision="fp32",
            patience=20, cv_folds=5, objectives={"binding":"spcc","classification":"mcc"},
            sampler="TPESampler(seed=run_seed)", model_seed="run_seed",
            target_model_revision="c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
            package_versions={name:version(name) for name in ("torch","numpy","scipy","rdkit","lightgbm","autopeptideml","optuna","transformers")},
            protocol="IBM YAML, explicit objectives and RNG seeds; not byte-identical historical defaults",
            benchmark_commit=subprocess.check_output(["git","-C",paths["ibm_root"],"rev-parse","HEAD"],text=True).strip())
        inputs = [Path(paths["grover_checkpoint"]), Path(paths["projection_checkpoint"])]
        inputs += list((Path(paths["ibm_root"])/"downstream_data").glob("*.csv"))
        inputs += list((Path(paths["ibm_root"])/"rep_transfer/h_param_search").glob("lightgbm_*.yml"))
        inputs += [Path(__file__), ROOT/"scripts/export_ibm_features.py", ROOT/"scripts/run_ibm_evaluation.py"]
        inputs += [Path(paths["ibm_root"])/"rep_transfer/evaluation_ood.py"]
        inputs += list((ROOT/"src/pepclip").rglob("*.py"))
        snapshot = ROOT/".cache/huggingface/hub/models--facebook--esm2_t6_8M_UR50D/snapshots"/config["target_model_revision"]
        if not snapshot.is_dir():
            raise ValueError("Download the pinned ESM2-8M target model before launching offline workers")
        inputs += [f for f in snapshot.iterdir() if f.is_file()]
        if args.feature_run:
            config["feature_run"] = str(args.feature_run.resolve())
            selected = "grover-pepclip-concat" if args.representation == "grover-and-concat" else args.representation
            inputs += cached_inputs(args.feature_run.resolve(), Path(paths["ibm_root"]), selected)
        if args.representation == "grover-and-concat":
            config["representations"] = ["grover-base", "grover-pepclip-concat"]
        config["concatenation"] = "raw GROVER 1600 then cached L2-normalized PepCLIP 2048; no further scaling; binding target 320 appended once"
        config["input_hashes"] = {str(f):sha(f) for f in inputs}
        if args.reuse_grover_run:
            reuse_evaluations(args.reuse_grover_run.resolve(), run, config)
        write_json(run / "config.json", config)
    for filename, digest in config["input_hashes"].items():
        if sha(filename) != digest:
            raise ValueError(f"Resume input changed: {filename}")
    for name, expected in config.get("package_versions", {}).items():
        if version(name) != expected:
            raise ValueError(f"Resume dependency changed: {name}")
    jobs = job_specs(config["seeds"])
    if config.get("feature_run"):
        jobs = {job: [] for job in jobs if job.startswith("eval-")}
    if config.get("representations"):
        jobs = evaluation_jobs(config["seeds"], config["representations"])
    done = {job: data for job in jobs if (data := completed(run, job)) is not None}
    for rep in config.get("representations", [config["representation"]]):
        aggregate(run, done, rep)
    running, failed = {}, {}
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), HF_HOME=str(ROOT/".cache/huggingface"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLCONFIGDIR=str(ROOT/".cache/matplotlib"))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(config["threads"])
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def status(state):
        write_json(run/"status.json", dict(state=state, pid=os.getpid(), total=len(jobs),
            completed=len(done), evaluations_completed=sum(j.startswith("eval-") for j in done),
            running={j:dict(pid=v[0].pid,attempt=str(v[2].relative_to(run))) for j,v in running.items()},
            failed=failed, updated=time.time(), workers=config["workers"], threads_per_worker=config["threads"]))
    try:
        while len(done) < len(jobs):
            for job,(process,log,attempt) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del running[job]
                    if code != 0:
                        failed[job] = dict(exit_code=code, log=str(attempt/"worker.log"))
                    else:
                        data = completed(run, job)
                        if data is None:
                            failed[job] = dict(error="Worker exited without completion marker")
                        else:
                            done[job] = data
                            for rep in config.get("representations", [config["representation"]]):
                                aggregate(run, done, rep)
            if failed:
                raise RuntimeError(f"Jobs failed: {failed}")
            for job,deps in jobs.items():
                if len(running) >= config["workers"]:
                    break
                if job in done or job in running or not all(d in done for d in deps):
                    continue
                folder = run/"jobs"/job
                folder.mkdir(parents=True, exist_ok=True)
                number = len(list(folder.glob("attempt-*")))+1
                attempt = folder/f"attempt-{number:03d}"
                attempt.mkdir()
                log = (attempt/"worker.log").open("w")
                process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--run",str(run),
                    "--worker",job,"--attempt",str(attempt)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                running[job] = (process,log,attempt)
            status("running")
            if len(done) < len(jobs):
                time.sleep(2)
        status("complete")
    except BaseException:
        for process,log,_ in running.values():
            if process.poll() is None:
                process.terminate()
        for process,log,_ in running.values():
            process.wait()
            log.close()
        running.clear()
        status("failed" if failed else "interrupted")
        raise


if __name__ == "__main__":
    main()
