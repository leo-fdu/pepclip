"""Export official IBM rows to raw GROVER and projected PepCLIP features.

No sequence reconstruction, length filtering, deduplication, or split changes.
Partial smoke exports cannot be written to the official reps directory.
"""
import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

DATASETS = tuple(f"{kind}-{task}" for kind in ("c", "nc")
                 for task in ("binding", "cpp", "antibacterial", "antiviral"))


def read_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"SMILES", "labels"} <= rows[0].keys():
        raise ValueError(f"Missing SMILES/labels or empty CSV: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibm-root", type=Path, required=True)
    parser.add_argument("--grover-source", type=Path, required=True)
    parser.add_argument("--grover-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "cuda-fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; not benchmark features")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.limit < 0:
        parser.error("batch-size/threads must be positive; limit must be nonnegative")
    if args.precision == "cuda-fp16" and args.device != "cuda":
        parser.error("cuda-fp16 requires --device cuda")
    if args.limit and args.output.resolve() == (args.ibm_root / "reps").resolve():
        parser.error("Smoke exports must not populate official reps/")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    sys.path.insert(0, str(args.grover_source.resolve()))
    import numpy as np
    import torch
    from rdkit import Chem
    from pepclip.encoders.grover import GroverEncoder
    from pepclip.inference import ProjectedEmbeddingEncoder
    from pepclip.utils import file_sha256

    torch.set_num_threads(args.threads)
    print(json.dumps(dict(event="loading_models", device=args.device)), flush=True)
    load_started = time.monotonic()
    projection = ProjectedEmbeddingEncoder(args.projection_checkpoint, side="b", device=args.device)
    encoder = GroverEncoder(args.grover_checkpoint, device=args.device, trusted_checkpoint=True)
    expected = projection.source_metadata["provenance"]["encoder_metadata"]
    actual = encoder.metadata()
    # Runtime precision and library versions may differ on CPU; record them below.
    for key in ("checkpoint_sha256", "output_dim", "pooling", "extra_features", "source_sha256"):
        if actual[key] != expected[key]:
            raise ValueError(f"GROVER differs from alignment training: {key}")
    args.output.mkdir(parents=True)
    report = dict(status="running", smoke=bool(args.limit), device=args.device,
                  model_load_seconds=time.monotonic() - load_started,
                  precision=args.precision, threads=args.threads,
                  projection_sha256=file_sha256(args.projection_checkpoint),
                  encoder=actual, training_encoder=expected,
                  training_precision=projection.source_metadata["provenance"].get("precision"),
                  benchmark_commit=subprocess.check_output(
                      ["git", "-C", str(args.ibm_root), "rev-parse", "HEAD"], text=True).strip(),
                  datasets={})
    report_path = args.output / "export.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(event="models_ready", seconds=report["model_load_seconds"])), flush=True)
    for dataset in args.datasets:
        source = args.ibm_root / "downstream_data" / f"{dataset}.csv"
        rows = read_rows(source)
        total = len(rows)
        rows = rows[:args.limit] if args.limit else rows
        invalid = [i for i, row in enumerate(rows)
                   if not row["SMILES"].strip() or Chem.MolFromSmiles(row["SMILES"]) is None]
        if invalid:
            report.update(status="invalid_structures", failed_dataset=dataset, invalid_rows=invalid)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            raise ValueError(f"{dataset}: invalid SMILES at zero-based rows {invalid}; no rows dropped")
        raw, aligned, batches = [], [], []
        started = time.monotonic()
        for start in range(0, len(rows), args.batch_size):
            smiles = [r["SMILES"] for r in rows[start:start + args.batch_size]]
            context = torch.autocast("cuda", dtype=torch.float16) if args.precision == "cuda-fp16" else nullcontext()
            if args.device == "cuda":
                torch.cuda.synchronize()
            batch_started = time.monotonic()
            with context:
                features = encoder.encode(smiles)
            graph_seconds = time.monotonic() - batch_started
            projection_started = time.monotonic()
            projected = projection.encode(features)
            projection_seconds = time.monotonic() - projection_started
            batches.append(dict(rows=len(smiles), graph_seconds=graph_seconds,
                                projection_seconds=projection_seconds))
            if not torch.isfinite(features).all() or not torch.isfinite(projected).all():
                raise ValueError(f"Nonfinite features: {dataset}, row {start}")
            raw.append(features.numpy())
            aligned.append(projected.numpy())
            print(json.dumps(dict(dataset=dataset, rows=min(start + args.batch_size, len(rows)),
                                  total=len(rows))), flush=True)
        arrays = {"grover-base": np.concatenate(raw), "pepclip": np.concatenate(aligned)}
        files = {}
        for name, array in arrays.items():
            path = args.output / f"{name}_{dataset}.pickle"
            with path.open("xb") as handle:
                pickle.dump(array, handle, protocol=4)
            files[name] = dict(shape=list(array.shape), sha256=file_sha256(path))
        # CSV order preserves duplicates and protein-peptide pairs.
        report["datasets"][dataset] = dict(source_sha256=file_sha256(source),
            source_rows=total, exported_rows=len(rows), row_order="original CSV, zero-based",
            files=files, seconds=time.monotonic() - started, batches=batches,
            atom_counts=[Chem.MolFromSmiles(r["SMILES"]).GetNumAtoms() for r in rows],
            sequence_lengths=[len(r["sequence"]) if r.get("sequence") else None for r in rows])
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
