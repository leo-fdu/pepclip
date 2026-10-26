"""Resumable four-GPU extraction with a bounded, lossless CPU graph spool.

Run on Linux/CUDA. Input shards and output rows retain FASTA order; sorting
by length is restricted to execution order inside each shard. No splits made.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from pepclip.data import AMINO_ACIDS, GRAPH_POLICY, read_fasta, sequence_to_smiles
from pepclip.utils import file_sha256, fingerprint


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def terminate_requested(signum, frame):
    raise SystemExit(128 + signum)


@contextmanager
def exclusive(path, blocking=True):
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def emit(event, **fields):
    print(json.dumps(dict(time=utc(), event=event, pid=os.getpid(), **fields)), flush=True)


def spool_size(directory):
    total = 0
    for path in directory.glob("*.zip"):
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            # A GPU may finish and unlink its committed input after globbing.
            continue
    return total


def prepare(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}
    # Reject changed models, source, precision, batch/shard policy on resume.
    identity = dict(settings=settings, source_hashes={
        "extractor": file_sha256(__file__),
        "fast_graph": file_sha256(Path(__file__).resolve().parents[1] / "src/pepclip/encoders/fast_graph.py"),
        "fasta": file_sha256(args.fasta),
        "esm_weights": file_sha256(args.esm_checkpoint),
        "grover_weights": file_sha256(args.grover_checkpoint)})
    source_root = Path(__file__).resolve().parents[1] / "src/pepclip"
    identity["source_hashes"]["pepclip"] = {
        str(p.relative_to(source_root)): file_sha256(p) for p in sorted(source_root.rglob("*.py"))}
    if identity["source_hashes"]["fasta"] != args.fasta_sha256:
        raise ValueError("FASTA checksum differs from the selected dataset")
    meta_path = root / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["identity"] != identity:
            raise ValueError("Run settings/source changed; use a different output directory")
        return meta
    for folder in ("input-shards", "graph-spool", "claims", "progress", "telemetry", "esm2", "grover"):
        (root / folder).mkdir(exist_ok=True)
    for model in ("esm2", "grover"):
        (root / "progress" / model).mkdir(exist_ok=True)
    shards, rows, count, ids = [], [], 0, set()
    ids_tmp = root / "ids.json.tmp"
    with ids_tmp.open("w") as handle:
        handle.write("[")
        for header, sequence in read_fasta(args.fasta):
            sid = header.split()[0]
            if sid in ids or not 10 <= len(sequence) <= 100 or set(sequence) - AMINO_ACIDS:
                raise ValueError(f"Invalid or duplicate FASTA entry: {sid}")
            ids.add(sid)
            if count:
                handle.write(",")
            handle.write(json.dumps(sid))
            rows.append(dict(row=count, sample_id=sid, sequence=sequence))
            count += 1
            if len(rows) == args.shard_size:
                name = f"{len(shards):06d}"
                atomic_json(root / "input-shards" / (name + ".json"), rows)
                shards.append(dict(name=name, start=count-len(rows), count=len(rows)))
                rows = []
        if rows:
            name = f"{len(shards):06d}"
            atomic_json(root / "input-shards" / (name + ".json"), rows)
            shards.append(dict(name=name, start=count-len(rows), count=len(rows)))
        handle.write("]\n")
        handle.flush()
        os.fsync(handle.fileno())
    if count != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} samples, found {count}")
    os.replace(ids_tmp, root / "ids.json")
    import numpy as np
    for model, dim in (("esm2", args.esm_dim), ("grover", 1600)):
        path = root / model / "features.npy"
        # run.json is the initialization commit; interruption before it is safe
        # to recreate, because no GPU workers have been launched yet.
        values = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(count, dim))
        values.flush()
        del values
    meta = dict(identity=identity, started_utc=utc(), count=count, shards=shards,
                graph_policy=GRAPH_POLICY, dataset_release="unknown",
                duplicates="Unique IDs retained; sequence deduplication required when constructing training pairs",
                source_git_revision=args.git_revision)
    atomic_json(meta_path, meta)
    emit("prepared", samples=count, shards=len(shards))
    return meta


def init_cpu(source):
    sys.path.insert(0, source)
    import torch
    from rdkit import RDLogger
    from pepclip.encoders.fast_graph import fast_molgraph_class
    from grover.data.molgraph import BatchMolGraph
    from argparse import Namespace
    global FastGraph, BatchGraph, graph_args
    RDLogger.DisableLog("rdApp.warning")
    torch.set_num_threads(1)
    FastGraph, BatchGraph, graph_args = fast_molgraph_class(), BatchMolGraph, Namespace(bond_drop_rate=0)


def build_shard(payload):
    root, name, batch_size = payload
    root = Path(root)
    if (root / "progress/grover" / (name + ".json")).exists():
        return name
    target = root / "graph-spool" / (name + ".zip")
    if target.exists():
        return name
    from pepclip.encoders.fast_graph import pack_graph, unpack_graph
    import torch
    rows = sorted(json.loads((root / "input-shards" / (name + ".json")).read_text()),
                  key=lambda row: (len(row["sequence"]), row["row"]))
    temp = target.with_suffix(f".{os.getpid()}.tmp")
    started = time.monotonic()
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("rows.json", json.dumps([r["row"] for r in rows]))
        for offset in range(0, len(rows), batch_size):
            graphs = [FastGraph(sequence_to_smiles(r["sequence"]), graph_args)
                      for r in rows[offset:offset+batch_size]]
            components = BatchGraph(graphs, graph_args).get_components()
            blob = pack_graph(components)
            # Verify every batch before committing its spool shard.
            restored = unpack_graph(blob)
            if not all(torch.equal(a, b) for a, b in zip(components, restored)):
                raise ValueError("Lossless graph roundtrip failed")
            archive.writestr(f"{offset:06d}.npz", blob)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, target)
    return dict(name=name, samples=len(rows), seconds=time.monotonic()-started, bytes=target.stat().st_size)


def cpu_producer(args, meta):
    os.setpgrp()
    root = args.output
    pending = [s["name"] for s in meta["shards"] if
               not (root / "progress/grover" / (s["name"] + ".json")).exists()
               and not (root / "graph-spool" / (s["name"] + ".zip")).exists()]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.cpu_workers, mp_context=ctx,
            initializer=init_cpu, initargs=(str(args.grover_source),)) as pool:
        futures, cursor, completed, total_samples = {}, 0, 0, 0
        while cursor < len(pending) or futures:
            spool_bytes = spool_size(root / "graph-spool")
            while cursor < len(pending) and len(futures) < args.cpu_workers * 2 and spool_bytes < args.spool_gb * 10**9:
                name = pending[cursor]
                futures[pool.submit(build_shard, (str(root), name, args.grover_batch))] = name
                cursor += 1
            if not futures:
                time.sleep(2)
                continue
            done, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                del futures[future]
                completed += 1
                if isinstance(result, dict):
                    total_samples += result["samples"]
                atomic_json(root / "cpu-status.json", dict(time=utc(), completed_shards=completed,
                    total_shards=len(pending), samples_prepared_this_session=total_samples, latest=result))
    emit("cpu_complete")


def claim_job(root, model, shards):
    for shard in shards:
        name = shard["name"]
        if (root / "progress" / model / (name + ".json")).exists():
            continue
        if model == "grover" and not (root / "graph-spool" / (name + ".zip")).exists():
            continue
        claim = root / "claims" / f"{model}-{name}"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        # A competing finisher may have committed since the first check.
        if (root / "progress" / model / (name + ".json")).exists():
            claim.rmdir()
            continue
        return shard, claim
    return None


def load_graph_batches(path):
    from pepclip.encoders.fast_graph import unpack_graph
    with zipfile.ZipFile(path) as archive:
        rows = json.loads(archive.read("rows.json"))
        for name in sorted(n for n in archive.namelist() if n.endswith(".npz")):
            yield unpack_graph(archive.read(name)), rows


def gpu_worker(args, meta, gpu):
    os.setpgrp()
    import numpy as np
    import torch
    from pepclip.encoders.esm2 import ESM2Encoder
    from pepclip.encoders.grover import GroverEncoder
    torch.set_num_threads(args.gpu_cpu_threads)
    torch.cuda.set_device(gpu)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, encoders = args.output, {}
    primary = "esm2" if gpu < args.esm_gpus else "grover"
    matrices = {model: np.load(root / model / "features.npy", mmap_mode="r+") for model in ("esm2", "grover")}
    while True:
        selected = None
        for model in (primary, "grover" if primary == "esm2" else "esm2"):
            selected = claim_job(root, model, meta["shards"])
            if selected:
                break
        if selected is None:
            if all(len(list((root / "progress" / m).glob("*.json"))) == len(meta["shards"])
                   for m in ("esm2", "grover")):
                break
            time.sleep(1)
            continue
        shard, claim = selected
        if model not in encoders:
            # Serialize CPU checkpoint loading to bound peak host RAM.
            with exclusive(root / "model-load.lock"):
                encoder = (ESM2Encoder(model_name=args.esm_model, checkpoint=str(args.esm_checkpoint),
                           device=f"cuda:{gpu}") if model == "esm2" else
                           GroverEncoder(str(args.grover_checkpoint), device=f"cuda:{gpu}", trusted_checkpoint=True))
            if encoder.model.training or any(p.requires_grad for p in encoder.model.parameters()):
                raise ValueError("Encoder is not frozen in eval mode")
            encoders[model] = encoder
            atomic_json(root / f"encoder-{model}-{gpu}.json", encoder.metadata())
            emit("model_loaded", gpu=gpu, model=model)
        encoder = encoders[model]
        started = time.monotonic()
        values = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            if model == "esm2":
                rows = sorted(json.loads((root / "input-shards" / (shard["name"] + ".json")).read_text()),
                              key=lambda r: (len(r["sequence"]), r["row"]))
                indices = [r["row"] for r in rows]
                for offset in range(0, len(rows), args.esm_batch):
                    selected_rows = rows[offset:offset+args.esm_batch]
                    values.append(encoder.encode([r["sequence"] for r in selected_rows]).numpy())
            else:
                # Decode the next packed batch while the current CUDA kernels execute.
                with ThreadPoolExecutor(1) as prefetch:
                    iterator = iter(load_graph_batches(root / "graph-spool" / (shard["name"] + ".zip")))
                    next_batch = prefetch.submit(next, iterator, None)
                    while True:
                        item = next_batch.result()
                        if item is None:
                            break
                        components, indices = item
                        next_batch = prefetch.submit(next, iterator, None)
                        count = len(components[5])
                        values.append(encoder.model(components, [None] * count).detach().cpu().float().numpy())
        array = np.concatenate(values)
        if array.shape != (shard["count"], matrices[model].shape[1]) or not np.isfinite(array).all():
            raise ValueError("Invalid embedding shape or nonfinite output")
        # Sort back to contiguous source order before writing disjoint row ranges.
        order = np.argsort(indices)
        if not np.array_equal(np.asarray(indices)[order], np.arange(shard["start"], shard["start"]+shard["count"])):
            raise ValueError("Shard identity/row permutation mismatch")
        array = array[order]
        matrices[model][shard["start"]:shard["start"]+shard["count"]] = array
        matrices[model].flush()
        with (root / model / "features.npy").open("rb") as handle:
            os.fsync(handle.fileno())
        result = dict(time=utc(), model=model, gpu=gpu, **shard, seconds=time.monotonic()-started,
                      sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        atomic_json(root / "progress" / model / (shard["name"] + ".json"), result)
        if model == "grover":
            (root / "graph-spool" / (shard["name"] + ".zip")).unlink(missing_ok=True)
        claim.rmdir()
        emit("shard_complete", **{k:v for k,v in result.items() if k != "time"})
    emit("gpu_complete", gpu=gpu)


def finalize_model(args, meta, model):
    import numpy as np
    # One model can be finalized/transferred while the other still computes.
    for model in (model,):
        bundle = args.output / model
        if (bundle / "metadata.json").exists():
            continue
        array = np.load(bundle / "features.npy", mmap_mode="r")
        for shard in meta["shards"]:
            record = json.loads((args.output / "progress" / model / (shard["name"] + ".json")).read_text())
            chunk = array[shard["start"]:shard["start"]+shard["count"]]
            if not np.isfinite(chunk).all() or hashlib.sha256(chunk.tobytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Output checksum mismatch: {model}/{shard['name']}")
        info_path = next(args.output.glob(f"encoder-{model}-*.json"))
        encoder = json.loads(info_path.read_text())
        provenance = dict(encoder=args.esm_model if model == "esm2" else "grover-base",
            version="fair-esm-2.0.0" if model == "esm2" else args.grover_commit,
            pooling=encoder["pooling"], preprocessing="canonical 10-100 aa; " + GRAPH_POLICY,
            precision="autocast_fp16; FP32 model weights; pooled outputs stored as FP32",
            normalization="none", encoder_metadata=encoder, run_identity=meta["identity"],
            graph_optimization="exact bond traversal; lossless packed features" if model == "grover" else "none",
            source_dataset_version="unknown", splits="not assigned by extraction")
        shutil.copyfile(args.output / "ids.json", bundle / "ids.json")
        atomic_json(bundle / "metadata.json", dict(schema_version=2, shape=list(array.shape),
            dtype=str(array.dtype), provenance=provenance, features_sha256=file_sha256(bundle / "features.npy"),
            ids_sha256=file_sha256(bundle / "ids.json")))
        emit("bundle_complete", model=model, shape=list(array.shape))


def main():
    signal.signal(signal.SIGTERM, terminate_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fasta", "output", "grover-source", "esm-checkpoint", "grover-checkpoint"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fasta-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--esm-model", default="esm2_t36_3B_UR50D")
    parser.add_argument("--esm-dim", type=int, default=2560)
    parser.add_argument("--grover-commit", default="40b6d97098e4508687912f3c05eca369fc2c6213")
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--cpu-workers", type=int, default=40)
    parser.add_argument("--gpu-cpu-threads", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--esm-gpus", type=int, default=2)
    parser.add_argument("--esm-batch", type=int, default=32)
    parser.add_argument("--grover-batch", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--spool-gb", type=float, default=20)
    args = parser.parse_args()
    for name in ("fasta", "output", "grover_source", "esm_checkpoint", "grover_checkpoint"):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.cpu_workers, args.gpu_cpu_threads, args.gpus, args.esm_batch,
           args.grover_batch, args.shard_size, args.expected_count, args.spool_gb) < 1:
        raise ValueError("Sizes/workers must be positive")
    if not 0 <= args.esm_gpus <= args.gpus:
        raise ValueError("Invalid ESM GPU allocation")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive(args.output / "run.lock", blocking=False):
        meta = prepare(args)
        if (args.output / "complete.json").exists():
            emit("already_complete")
            return
        # Only an exclusive supervisor may clear uncommitted claims on restart.
        for path in (args.output / "claims").iterdir():
            path.rmdir()
        sys.path.insert(0, str(args.grover_source))
        ctx = mp.get_context("spawn")
        processes = [ctx.Process(target=cpu_producer, args=(args, meta))]
        processes += [ctx.Process(target=gpu_worker, args=(args, meta, gpu)) for gpu in range(args.gpus)]
        finalizers = ThreadPoolExecutor(2)
        finalized = {}
        previous_ticks = None
        try:
            for process in processes:
                process.start()
            while any(p.is_alive() for p in processes):
                failed = [p for p in processes if p.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Worker failed: {[(p.pid,p.exitcode) for p in failed]}")
                counts = {m: len(list((args.output / "progress" / m).glob("*.json"))) for m in ("esm2", "grover")}
                for model, count in counts.items():
                    if count == len(meta["shards"]) and model not in finalized:
                        finalized[model] = finalizers.submit(finalize_model, args, meta, model)
                for task in finalized.values():
                    if task.done():
                        task.result()
                ticks = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
                cpu_usage = None
                if previous_ticks:
                    delta = [a-b for a,b in zip(ticks,previous_ticks)]
                    total = sum(delta)
                    if total:
                        cpu_usage = dict(busy_percent=100*(total-delta[3]-delta[4]-delta[7])/total,
                                         iowait_percent=100*delta[4]/total, steal_percent=100*delta[7]/total)
                previous_ticks = ticks
                telemetry = dict(time=utc(), completed_shards=counts, total_shards=len(meta["shards"]),
                    cpu_interval=cpu_usage, loadavg=Path("/proc/loadavg").read_text().strip(),
                    free_disk_bytes=shutil.disk_usage(args.output).free,
                    gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv,noheader"], text=True),
                    cpu=subprocess.check_output(["ps", "-eo", "pid,ppid,pcpu,rss,comm"], text=True))
                atomic_json(args.output / "status.json", telemetry)
                with (args.output / "telemetry/history.jsonl").open("a") as handle:
                    handle.write(json.dumps(telemetry)+"\n")
                emit("status", completed_shards=counts, total_shards=len(meta["shards"]))
                time.sleep(30)
            for p in processes:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"Worker {p.pid} failed with {p.exitcode}")
            for model in ("esm2", "grover"):
                if model not in finalized:
                    finalized[model] = finalizers.submit(finalize_model, args, meta, model)
            for task in finalized.values():
                task.result()
            atomic_json(args.output / "complete.json", dict(time=utc(), samples=meta["count"],
                models=["esm2", "grover"], run_fingerprint=fingerprint(meta["identity"])))
        except BaseException:
            for p in processes:
                if p.is_alive():
                    # CPU pool children share the producer's process group.
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        p.terminate()
            for p in processes:
                if p.pid is not None:
                    p.join(timeout=10)
                    # Also reap a pool child born during producer termination.
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        if p.is_alive():
                            p.kill()
                    p.join()
            raise
        finally:
            finalizers.shutdown(wait=True)


if __name__ == "__main__":
    main()
