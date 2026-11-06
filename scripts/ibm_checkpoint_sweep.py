"""Configuration-driven IBM checkpoint comparisons; dry-run never writes outputs."""
import argparse
import fcntl
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
from ibm_ood_queue import ROOT, TASKS, sha, write_json

REPRESENTATIONS = ("pepclip", "grover-base", "grover-pepclip-concat")
PATH_KEYS = ("run", "feature_source", "ibm_root", "grover_source", "grover_checkpoint")

def load_config(path):
    config = json.loads(Path(path).read_text())
    required = set(PATH_KEYS) | {"checkpoints", "representations", "tasks", "seeds",
                                "workers", "threads_per_worker", "hpo_trials"}
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    for key in PATH_KEYS:
        config[key] = str((ROOT / config[key]).resolve())
    for key, allowed in (("representations", REPRESENTATIONS), ("tasks", TASKS)):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f"Invalid {key}")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique nonnegative uint32 integers")
    for key in ("workers", "threads_per_worker", "hpo_trials"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not isinstance(config["checkpoints"], list) or not config["checkpoints"]:
        raise ValueError("At least one checkpoint required")
    names = set()
    for checkpoint in config["checkpoints"]:
        if set(checkpoint) != {"name", "path", "epoch"}:
            raise ValueError("Checkpoint requires name, path, epoch")
        name = checkpoint["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Checkpoint names must be unique safe directory names")
        names.add(name)
        if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 1:
            raise ValueError("Invalid checkpoint epoch")
        checkpoint["path"] = str((ROOT / checkpoint["path"]).resolve())
    return config

def commands(config):
    root = Path(config["run"])
    for cp in config["checkpoints"]:
        features = root / f"features-{cp['name']}"
        prepare = [sys.executable, str(ROOT / "scripts/reproject_ibm_features.py"),
                   "--source", config["feature_source"], "--output", str(features),
                   "--checkpoint", cp["path"], "--epoch", str(cp["epoch"]),
                   "--ibm-root", config["ibm_root"], "--threads", str(config["threads_per_worker"])]
        run = root / cp["name"]
        evaluate = [sys.executable, str(ROOT / "scripts/ibm_ood_queue.py"), "--run", str(run),
                    "--feature-run", str(features), "--representations", *config["representations"],
                    "--tasks", *config["tasks"], "--seed-ids", *map(str, config["seeds"]),
                    "--ibm-root", config["ibm_root"], "--grover-source", config["grover_source"],
                    "--grover-checkpoint", config["grover_checkpoint"],
                    "--projection-checkpoint", cp["path"],
                    "--workers", str(config["workers"]), "--threads", str(config["threads_per_worker"]),
                    "--trials", str(config["hpo_trials"])]
        yield cp, run, prepare, evaluate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    total = len(config["checkpoints"]) * len(config["representations"]) * len(config["tasks"]) * len(config["seeds"])
    if args.dry_run:
        print(json.dumps(dict(total_evaluations=total, config=config,
                              commands=[dict(prepare=p, evaluate=e) for _, _, p, e in commands(config)]), indent=2))
        return
    manifest = dict(schema=2, config=config, total_evaluations=total,
                    checkpoint_hashes={cp["name"]: sha(Path(cp["path"])) for cp in config["checkpoints"]})
    root = Path(config["run"])
    # Reject incompatible legacy manifests without changing their files or caches.
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Sweep configuration/checkpoints changed; use a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "sweep.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Sweep manifest changed")
    write_json(path, manifest)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(config["threads_per_worker"])
    try:
        for cp, run, prepare, evaluate in commands(config):
            write_json(root / "status.json", dict(state="preparing", checkpoint=cp["name"], pid=os.getpid()))
            subprocess.run(prepare, cwd=ROOT, env=env, check=True)
            if args.prepare_only:
                continue
            if (run / "config.json").exists():
                evaluate = [sys.executable, str(ROOT / "scripts/ibm_ood_queue.py"), "--run", str(run), "--resume"]
            write_json(root / "status.json", dict(state="running", checkpoint=cp["name"], pid=os.getpid()))
            caffeine = shutil.which("agent-caffeine")
            if caffeine:
                evaluate = [caffeine, "hold", "--reason", f"PepCLIP {cp['name']} IBM benchmark", "--", *evaluate]
            else:
                print("Agent Caffeine unavailable; sleep protection not enabled", flush=True)
            subprocess.run(evaluate, cwd=ROOT, env=env, check=True)
        write_json(root / "status.json", dict(state="prepared" if args.prepare_only else "complete", pid=os.getpid()))
    except BaseException:
        write_json(root / "status.json", dict(state="failed", pid=os.getpid()))
        raise

if __name__ == "__main__":
    main()
