"""Measure real frozen ESM2/GROVER extraction on a representative peptide sample."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from pepclip.data import sequence_to_smiles
from pepclip.encoders.esm2 import ESM2Encoder
from pepclip.encoders.grover import GroverEncoder
from pepclip.utils import file_sha256, write_json


def precision_context(name):
    return torch.autocast("cuda", dtype=torch.float16) if name == "amp_fp16" else nullcontext()


def measure(encoder, inputs, args, precision, batch_size, counters):
    for _ in range(args.warmup):
        with precision_context(precision):
            encoder.encode(inputs[:batch_size])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rates, times, graph_times, batch_times, write_times = [], [], [], [], []
    for repeat in range(args.repeats):
        counters.update(graph=0.0, batch=0.0)
        # Distinct windows of an unbiased source sample, cycling when necessary.
        selected = [inputs[(repeat * args.samples_per_repeat + i) % len(inputs)]
                    for i in range(args.samples_per_repeat)]
        path = args.output.parent / f"{args.model}-{precision}-b{batch_size}-r{repeat}.npy"
        dest = np.lib.format.open_memmap(path, mode="w+", dtype="float32",
                                        shape=(len(selected), encoder.output_dim))
        write_time = 0.0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for offset in range(0, len(selected), batch_size):
            with precision_context(precision):
                output = encoder.encode(selected[offset:offset + batch_size])
            if not torch.isfinite(output).all():
                raise ValueError("Nonfinite real encoder output")
            begin_write = time.perf_counter()
            dest[offset:offset + len(output)] = output.numpy()
            write_time += time.perf_counter() - begin_write
        begin_write = time.perf_counter()
        dest.flush()
        write_time += time.perf_counter() - begin_write
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        rates.append(len(selected) / seconds)
        times.append(seconds)
        graph_times.append(counters["graph"])
        batch_times.append(counters["batch"])
        write_times.append(write_time)
        del dest
        path.unlink()
    return dict(status="ok", precision=precision, batch_size=batch_size,
        sample_count_per_repeat=args.samples_per_repeat,
        median_samples_per_second=statistics.median(rates), samples_per_second_repeats=rates,
        elapsed_seconds_repeats=times, cpu_graph_seconds_repeats=graph_times,
        cpu_batch_seconds_repeats=batch_times, output_write_seconds_repeats=write_times,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["esm2", "grover"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "amp_fp16"], default=["fp32", "amp_fp16"])
    parser.add_argument("--samples-per-repeat", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.samples_per_repeat, args.repeats, args.warmup, args.threads, *args.batch_sizes) < 1:
        raise ValueError("Benchmark sizes and timing settings must be positive")
    if max(args.batch_sizes) > args.samples_per_repeat:
        raise ValueError("Use at least one full batch per repeat")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sample = json.loads(args.sample.read_text())
    if len(sample) < args.samples_per_repeat:
        raise ValueError("Sample is too small")
    sequences = [row["sequence"] for row in sample]
    result = dict(status="running", model=args.model,
        started_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        torch=str(torch.__version__), numpy=np.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), sample_sha256=file_sha256(args.sample),
        sample_count=len(sample), mean_sample_length=statistics.mean(map(len, sequences)),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cases=[], precision_comparison={},
        timing_scope="encode (tokenize/featurize, device transfer, forward, pooling, CPU float32 return), finite check, memmap write and flush; excludes loading/downloads",
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve().parents[1] / "src/pepclip/encoders" / (args.model + ".py")]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    counters = dict(graph=0.0, batch=0.0)
    start = time.perf_counter()
    if args.model == "esm2":
        encoder = ESM2Encoder(checkpoint=args.checkpoint, device="cuda")
        inputs = sequences
    else:
        encoder = GroverEncoder(args.checkpoint, device="cuda", trusted_checkpoint=True)
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.warning")
        preparation_start = time.perf_counter()
        inputs = [sequence_to_smiles(seq) for seq in sequences]
        preparation_seconds = time.perf_counter() - preparation_start
        result["sequence_to_smiles"] = dict(sample_count=len(inputs), seconds=preparation_seconds,
            samples_per_second=len(inputs) / preparation_seconds,
            mean_heavy_atoms=statistics.mean(Chem.MolFromSmiles(s).GetNumAtoms() for s in inputs))
        # Instrument the unchanged upstream constructors; no feature/graph changes.
        def timed_factory(factory, key):
            def wrapped(*a, **kw):
                start = time.perf_counter()
                value = factory(*a, **kw)
                counters[key] += time.perf_counter() - start
                return value
            return wrapped
        encoder.MolGraph = timed_factory(encoder.MolGraph, "graph")
        encoder.BatchMolGraph = timed_factory(encoder.BatchMolGraph, "batch")
    result["initialization_and_preparation_seconds"] = time.perf_counter() - start
    result["encoder_metadata"] = encoder.metadata()
    assert not encoder.model.training and all(not p.requires_grad for p in encoder.model.parameters())
    result["frozen_eval_verified"] = True
    comparison_inputs = inputs[:8]
    reference = encoder.encode(comparison_inputs)
    for precision in args.precisions:
        with precision_context(precision):
            compared = encoder.encode(comparison_inputs)
        result["precision_comparison"][precision] = dict(
            finite=bool(torch.isfinite(compared).all()),
            mean_cosine_to_fp32=F.cosine_similarity(reference, compared, dim=1).mean().item(),
            max_absolute_difference=(reference - compared).abs().max().item())
        del compared
        for batch_size in args.batch_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                case = measure(encoder, inputs, args, precision, batch_size, counters)
            except torch.OutOfMemoryError:
                case = dict(status="out_of_memory", precision=precision, batch_size=batch_size)
            result["cases"].append(case)
            write_json(args.output, result)
            print(json.dumps(case), flush=True)
    result.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output, result)


if __name__ == "__main__":
    main()
