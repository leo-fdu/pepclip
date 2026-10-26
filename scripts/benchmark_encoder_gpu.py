"""Separate prepared-input transfer/inference from resident-GPU inference."""
import argparse
from datetime import datetime, timezone
import gc
import io
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from benchmark_encoders import precision_context
from pepclip.data import sequence_to_smiles
from pepclip.encoders.esm2 import ESM2Encoder
from pepclip.encoders.grover import GroverEncoder
from pepclip.utils import file_sha256, write_json


def move_tensors(bundle):
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in bundle)


def packed_size_test(bundles, model, forward):
    """Lossless NPZ size/roundtrip; preprocessing and compression are not GPU time."""
    raw, compressed, compression_seconds, decoding_seconds = 0, 0, 0.0, 0.0
    for index, (bundle, count) in enumerate(bundles):
        arrays = {f"c{i}": value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.int64)
                  for i, value in enumerate(bundle)}
        raw += sum(value.nbytes for value in arrays.values())
        target = io.BytesIO()
        start = time.perf_counter()
        np.savez_compressed(target, **arrays)
        compression_seconds += time.perf_counter() - start
        payload = target.getvalue()
        compressed += len(payload)
        start = time.perf_counter()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            recovered = {key: archive[key] for key in arrays}
        decoding_seconds += time.perf_counter() - start
        assert all(np.array_equal(arrays[key], recovered[key]) for key in arrays)
        if index == 0 and model == "grover":
            restored = tuple(torch.from_numpy(recovered[f"c{i}"]) if isinstance(v, torch.Tensor)
                             else [tuple(row) for row in recovered[f"c{i}"].tolist()]
                             for i, v in enumerate(bundle))
            with torch.no_grad():
                expected = forward(bundle, count, "prepared_cpu", "fp32")
                actual = forward(restored, count, "prepared_cpu", "fp32")
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    return dict(raw_bytes=raw, npz_bytes=compressed, compression_seconds=compression_seconds,
                decoding_seconds=decoding_seconds, arrays_bitwise_roundtrip=True,
                graph_embedding_roundtrip_verified=model == "grover")


def run_case(encoder, cpu_batches, args, precision, mode):
    batches = [(move_tensors(bundle), n) for bundle, n in cpu_batches] if mode == "resident_gpu" else cpu_batches

    def forward(bundle, count, selected_mode, selected_precision):
        with torch.no_grad(), precision_context(selected_precision):
            if args.model == "esm2":
                tokens, lengths = bundle
                representations = encoder.model(tokens.cuda(), repr_layers=[encoder.layer],
                                                return_contacts=False)["representations"][encoder.layer]
                output = torch.stack([representations[i, 1:length + 1].mean(0)
                                      for i, length in enumerate(lengths)])
            else:
                output = encoder.model(bundle, [None] * count)
            return output.detach().cpu().float() if selected_mode == "prepared_cpu" else output

    for _ in range(args.warmup):
        forward(*batches[0], mode, precision)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rates, seconds, event_seconds = [], [], []
    last_output = None
    for _ in range(args.repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        begin.record()
        for bundle, count in batches:
            last_output = forward(bundle, count, mode, precision)
        end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        seconds.append(elapsed)
        event_seconds.append(begin.elapsed_time(end) / 1000)
        rates.append(args.samples / elapsed)
    if not torch.isfinite(last_output).all():
        raise ValueError("Nonfinite encoder output")
    case = dict(status="ok", precision=precision, mode=mode,
        median_samples_per_second=statistics.median(rates), samples_per_second_repeats=rates,
        wall_seconds_repeats=seconds, cuda_event_seconds_repeats=event_seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return case, forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["esm2", "grover"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "amp_fp16"], default=["fp32", "amp_fp16"])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.samples, args.repeats, args.warmup, *args.batch_sizes) < 1 or max(args.batch_sizes) > args.samples:
        raise ValueError("Invalid sample/batch/timing settings")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = json.loads(args.sample.read_text())[:args.samples]
    if len(rows) != args.samples:
        raise ValueError("Insufficient sample rows")
    encoder = (ESM2Encoder(checkpoint=args.checkpoint, device="cuda") if args.model == "esm2" else
               GroverEncoder(args.checkpoint, device="cuda", trusted_checkpoint=True))
    assert not encoder.model.training and all(not p.requires_grad for p in encoder.model.parameters())
    preparation_start = time.perf_counter()
    sequences = [r["sequence"] for r in rows]
    if args.model == "grover":
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.warning")
        graphs = [encoder.MolGraph(sequence_to_smiles(s), encoder.args) for s in sequences]
    result = dict(status="running", model=args.model, started_utc=datetime.now(timezone.utc).isoformat(),
        torch=str(torch.__version__), cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sample_sha256=file_sha256(args.sample), mean_timed_length=statistics.mean(map(len, sequences)),
        encoder_metadata=encoder.metadata(), cases=[], cache_size={},
        scope={"prepared_cpu":"Precomputed batched CPU inputs -> H2D + model + pooling + CPU FP32 output; no disk I/O",
               "resident_gpu":"Preloaded CUDA inputs -> model + pooling; output stays on GPU; no disk or transfer"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch_size in args.batch_sizes:
        start = time.perf_counter()
        cpu_batches = []
        for offset in range(0, len(rows), batch_size):
            if args.model == "esm2":
                selected = sequences[offset:offset + batch_size]
                _, _, tokens = encoder.converter([(str(i), s) for i, s in enumerate(selected)])
                bundle = (tokens, list(map(len, selected)))
                count = len(selected)
            else:
                selected = graphs[offset:offset + batch_size]
                bundle = encoder.BatchMolGraph(selected, encoder.args).get_components()
                count = len(selected)
            cpu_batches.append((bundle, count))
        result.setdefault("cpu_batch_preparation_seconds", {})[batch_size] = time.perf_counter() - start
        for precision in args.precisions:
            for mode in ("prepared_cpu", "resident_gpu"):
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    case, forward = run_case(encoder, cpu_batches, args, precision, mode)
                    case["batch_size"] = batch_size
                except torch.OutOfMemoryError:
                    case = dict(status="out_of_memory", precision=precision, mode=mode, batch_size=batch_size)
                result["cases"].append(case)
                write_json(args.output, result)
                print(json.dumps(case), flush=True)
        if args.model == "grover" and batch_size == args.batch_sizes[len(args.batch_sizes) // 2]:
            result["cache_size"] = packed_size_test(cpu_batches, args.model, forward)
            result["cache_size"]["sample_count"] = args.samples
        del cpu_batches
    result.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                  total_preparation_and_benchmark_seconds=time.perf_counter() - preparation_start)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
