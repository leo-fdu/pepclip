"""CPU-only sequence -> GROVER graph features -> lossless NPZ shard benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

from pepclip.data import sequence_to_smiles
from pepclip.utils import file_sha256, write_json


def initialize(source, barrier):
    global np, torch, MolGraph, BatchMolGraph, graph_args
    from argparse import Namespace
    sys.path.insert(0, source)
    import numpy as np
    import torch
    from rdkit import RDLogger
    from grover.data.molgraph import MolGraph, BatchMolGraph
    RDLogger.DisableLog("rdApp.warning")
    torch.set_num_threads(1)
    graph_args = Namespace(bond_drop_rate=0)
    graph = MolGraph(sequence_to_smiles("ACDEFGHIKL"), graph_args)
    BatchMolGraph([graph], graph_args).get_components()
    barrier.wait(timeout=120)


def ready(_):
    return os.getpid()


def process_chunk(payload):
    index, rows, directory = payload
    start = time.perf_counter()
    smiles = [sequence_to_smiles(row["sequence"]) for row in rows]
    after_smiles = time.perf_counter()
    graphs = [MolGraph(s, graph_args) for s in smiles]
    after_graphs = time.perf_counter()
    bundle = BatchMolGraph(graphs, graph_args).get_components()
    arrays = {f"c{i}": value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.int64)
              for i, value in enumerate(bundle)}
    after_batch = time.perf_counter()
    digest = hashlib.sha256()
    for key, value in arrays.items():
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes())
    path = Path(directory) / f"chunk-{index:06d}.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    after_write = time.perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        assert all(np.array_equal(value, archive[key]) for key, value in arrays.items())
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(chunk=index, samples=len(rows), sample_ids=[r["sample_id"] for r in rows],
        pid=os.getpid(), arrays_sha256=digest.hexdigest(),
        raw_bytes=sum(v.nbytes for v in arrays.values()), compressed_bytes=path.stat().st_size,
        smiles_seconds=after_smiles-start, graph_seconds=after_graphs-after_smiles,
        batch_seconds=after_batch-after_graphs, compression_and_write_seconds=after_write-after_batch,
        peak_worker_rss_mib=peak_rss/(2**20 if sys.platform=="darwin" else 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grover-source", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.samples, args.chunk_size, args.repeats, *args.workers) < 1:
        raise ValueError("Sizes must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    rows = json.loads(args.sample.read_text())[:args.samples]
    if len(rows) != args.samples:
        raise ValueError("Insufficient sampled sequences")
    source = str(args.grover_source.resolve())
    result = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        host_platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
        logical_cpus=os.cpu_count(), sample_sha256=file_sha256(args.sample),
        mean_timed_length=statistics.mean(len(r["sequence"]) for r in rows),
        source_sha256=file_sha256(Path(source)/"grover/data/molgraph.py"),
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope="sequence_to_smiles + unchanged GROVER MolGraph + BatchMolGraph + float32/int64 NPZ compression + fsync + exact decompression roundtrip + IPC; no models or GPUs",
        cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    ctx=mp.get_context("spawn")
    reference_digest=None
    for workers in args.workers:
        barrier=ctx.Barrier(workers)
        startup=time.perf_counter()
        case=dict(workers=workers,status="running",rates=[],repeats=[])
        result["cases"].append(case)
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize,
                                 initargs=(source,barrier)) as pool:
            list(pool.map(ready,range(workers)))
            case["startup_seconds"]=time.perf_counter()-startup
            for repeat in range(args.repeats):
                directory=args.output.parent/f"cpu-w{workers}-r{repeat}"
                directory.mkdir(exist_ok=False)
                chunks=[(i//args.chunk_size,rows[i:i+args.chunk_size],str(directory))
                        for i in range(0,len(rows),args.chunk_size)]
                start=time.perf_counter()
                items=list(pool.map(process_chunk,chunks))
                seconds=time.perf_counter()-start
                digest=items[0]["arrays_sha256"]
                if reference_digest is None:reference_digest=digest
                assert digest==reference_digest,"Worker count changed graph inputs"
                write_json(directory/"index.json",items)
                rate=len(rows)/seconds
                case["rates"].append(rate)
                case["repeats"].append(dict(seconds=seconds,raw_bytes=sum(x["raw_bytes"] for x in items),
                    compressed_bytes=sum(x["compressed_bytes"] for x in items),
                    worker_pids=sorted({x["pid"] for x in items}),
                    max_worker_rss_mib=max(x["peak_worker_rss_mib"] for x in items)))
                write_json(args.output,result)
                print(json.dumps(dict(workers=workers,repeat=repeat,samples_per_second=rate,seconds=seconds)),flush=True)
        case.update(status="complete",median_samples_per_second=statistics.median(case["rates"]))
        write_json(args.output,result)
    result.update(status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
                  first_chunk_arrays_sha256=reference_digest)
    write_json(args.output,result)


if __name__=="__main__":
    main()
