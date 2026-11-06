"""Run four checkpoint comparisons sequentially, sharing the existing IBM protocol."""
import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from ibm_ood_queue import ROOT, sha, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "sweep.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = ROOT / "outputs/ibm-pepclip-ood-20261029"
    training = ROOT / "outputs/mps-85-15-b512-20epochs"
    epochs = (5, 10, 15, 20)
    manifest = dict(epochs=list(epochs), seeds=list(range(5)), tasks=["antibacterial","antiviral","cpp","binding"],
                    representations=["pepclip","grover-pepclip-concat"], total_evaluations=160,
                    workers=5, threads=3, trials=100,
                    checkpoints={str(e): sha(training / f"epoch_{e:04d}.pt") for e in epochs})
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Sweep manifest changed")
    write_json(path, manifest)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[key] = "3"
    try:
        for epoch in epochs:
            checkpoint = training / f"epoch_{epoch:04d}.pt"
            features = root / f"features-epoch-{epoch:02d}"
            write_json(root / "status.json", dict(state="preparing", epoch=epoch, pid=os.getpid()))
            subprocess.run([sys.executable, str(ROOT / "scripts/reproject_ibm_features.py"),
                            "--source", str(source), "--output", str(features),
                            "--checkpoint", str(checkpoint), "--epoch", str(epoch),
                            "--ibm-root", str(ROOT / ".cache/ibm-peptide-benchmarks"), "--threads", "3"],
                           cwd=ROOT, env=env, check=True)
            if args.prepare_only:
                continue
            run = root / f"epoch-{epoch:02d}"
            command = [sys.executable, str(ROOT / "scripts/ibm_ood_queue.py"), "--run", str(run)]
            if (run / "config.json").exists():
                command += ["--resume"]
            else:
                command += ["--feature-run", str(features), "--representation", "pepclip-and-concat",
                            "--ibm-root", str(ROOT / ".cache/ibm-peptide-benchmarks"),
                            "--grover-source", str(ROOT / ".cache/encoder-benchmark/grover-source"),
                            "--grover-checkpoint", str(ROOT / ".cache/grover_base.pt"),
                            "--projection-checkpoint", str(checkpoint),
                            "--workers", "5", "--threads", "3", "--seeds", "5", "--trials", "100"]
            write_json(root / "status.json", dict(state="running", epoch=epoch, pid=os.getpid()))
            caffeine = shutil.which("agent-caffeine")
            if caffeine:
                command = [caffeine, "hold", "--reason", f"PepCLIP epoch {epoch} IBM benchmark", "--", *command]
            else:
                print("Agent Caffeine unavailable; sleep protection not enabled", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        write_json(root / "status.json", dict(state="prepared" if args.prepare_only else "complete", pid=os.getpid()))
    except BaseException:
        write_json(root / "status.json", dict(state="failed", pid=os.getpid()))
        raise

if __name__ == "__main__":
    main()
