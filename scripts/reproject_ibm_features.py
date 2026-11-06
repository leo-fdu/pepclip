"""Reproject validated IBM GROVER caches for one explicit epoch snapshot."""
import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import torch
from pepclip.inference import ProjectedEmbeddingEncoder
from pepclip.train import load_checkpoint
from ibm_ood_queue import cached_inputs, completed, job_specs, mark_done, sha, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ibm-root", type=Path, required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    inputs = cached_inputs(source, args.ibm_root, "grover-base")
    source_config = json.loads((source / "config.json").read_text())
    for filename, digest in source_config["input_hashes"].items():
        path = Path(filename)
        if path.suffix == ".csv" and sha(path) != digest:
            raise ValueError(f"Changed original CSV: {path}")
    state = load_checkpoint(args.checkpoint)
    if state["epoch"] != args.epoch:
        raise ValueError("Checkpoint epoch mismatch")
    del state
    torch.set_num_threads(args.threads)
    projection = ProjectedEmbeddingEncoder(args.checkpoint, side="b")
    expected = projection.source_metadata["provenance"]["encoder_metadata"]
    digest = sha(args.checkpoint)
    identity = dict(checkpoint=str(args.checkpoint.resolve()), projection_sha256=digest,
                    epoch=args.epoch, source=str(source),
                    source_hashes={str(f): sha(f) for f in inputs},
                    source_config_sha256=sha(source / "config.json"))
    if output.exists():
        if json.loads((output / "provenance.json").read_text()) != identity:
            raise ValueError("Existing projection cache provenance differs")
    else:
        output.mkdir(parents=True)
        write_json(output / "provenance.json", identity)
    for job in job_specs(0):
        if completed(output, job):
            continue
        old = completed(source, job)
        parent = output / "jobs" / job
        parent.mkdir(parents=True, exist_ok=True)
        attempt = parent / f"attempt-{len(list(parent.glob('attempt-*')))+1:03d}"
        attempt.mkdir()
        if job == "targets":
            files = []
            for name in old["files"]:
                target = attempt / Path(name).name
                shutil.copy2(source / name, target)
                files.append(target)
            mark_done(output, job, attempt, files)
            continue
        dataset = job.removeprefix("feature-")
        original = source / old["attempt"] / "reps"
        report = json.loads((original / "export.json").read_text())
        for key in ("checkpoint_sha256", "output_dim", "pooling", "extra_features", "source_sha256"):
            if report["encoder"][key] != expected[key]:
                raise ValueError(f"Training/cache encoder mismatch: {key}")
        folder = attempt / "reps"
        folder.mkdir()
        raw_path = original / f"grover-base_{dataset}.pickle"
        with raw_path.open("rb") as f:
            raw = np.asarray(pickle.load(f), dtype=np.float32)
        if raw.shape != (report["datasets"][dataset]["source_rows"], expected["output_dim"]):
            raise ValueError("Raw feature shape mismatch")
        aligned = np.concatenate([projection.encode(raw[i:i+256]).numpy()
                                  for i in range(0, len(raw), 256)])
        if not np.isfinite(aligned).all() or not np.allclose(np.linalg.norm(aligned, axis=1), 1, atol=1e-5):
            raise ValueError("Nonfinite or non-normalized projection")
        shutil.copy2(raw_path, folder / raw_path.name)
        target = folder / f"pepclip_{dataset}.pickle"
        with target.open("xb") as f:
            pickle.dump(aligned, f, protocol=4)
        entry = copy.deepcopy(report["datasets"][dataset])
        entry.pop("batches", None)
        entry.pop("seconds", None)
        entry["files"]["pepclip"] = dict(shape=list(aligned.shape), sha256=sha(target))
        result = dict(status="complete", smoke=False, device="cpu", precision="fp32",
                      projection_sha256=digest, checkpoint_epoch=args.epoch,
                      encoder=report["encoder"], training_encoder=expected,
                      benchmark_commit=report["benchmark_commit"],
                      source_export=str(original / "export.json"),
                      operation="projection-only; raw GROVER and targets copied unchanged",
                      datasets={dataset: entry})
        write_json(folder / "export.json", result)
        mark_done(output, job, attempt, list(folder.iterdir()))
        print(json.dumps(dict(epoch=args.epoch, dataset=dataset, rows=len(raw))), flush=True)
    cached_inputs(output, args.ibm_root, "grover-pepclip-concat")
    write_json(output / "status.json", dict(state="complete", epoch=args.epoch, projection_sha256=digest))

if __name__ == "__main__":
    main()
