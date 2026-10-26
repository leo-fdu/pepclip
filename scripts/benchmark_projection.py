"""Short CUDA benchmark of the actual projection model/loss; synthetic data only."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pepclip.cache import PairedFeatures
from pepclip.model import EmbeddingCLIP, clip_loss
from pepclip.train import ContrastiveBatches
from pepclip.utils import write_json


class SyntheticCache:
    def __init__(self, features, indices):
        self.features, self.indices = features, indices

    def get(self, sid):
        return torch.from_numpy(self.features[self.indices[sid]].copy())


def make_dataset(args):
    directory = args.output.parent / (args.output.stem + "-synthetic-cache")
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    arrays = []
    for side, dim in (("a", args.input_dim_a), ("b", args.input_dim_b)):
        array = np.lib.format.open_memmap(directory / f"{side}.npy", mode="w+",
                                        dtype="float32", shape=(args.samples, dim))
        for start in range(0, args.samples, 4096):
            end = min(start + 4096, args.samples)
            array[start:end] = rng.standard_normal((end - start, dim), dtype=np.float32)
        array.flush()
        arrays.append(array)
    rows = [dict(sample_id=f"synthetic-{i}", split="train") for i in range(args.samples)]
    indices = {row["sample_id"]: i for i, row in enumerate(rows)}
    return PairedFeatures(rows, SyntheticCache(arrays[0], indices),
                          SyntheticCache(arrays[1], indices), "train")


def run_case(args, batch_size, mode, dataset):
    torch.manual_seed(args.seed)
    model = EmbeddingCLIP(args.input_dim_a, args.input_dim_b,
                          args.hidden_dim, args.projection_dim).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader, iterator, resident = None, None, None
    if mode == "resident":
        resident = (torch.randn(batch_size, args.input_dim_a, device="cuda"),
                    torch.randn(batch_size, args.input_dim_b, device="cuda"))
    else:
        loader = DataLoader(dataset, batch_sampler=ContrastiveBatches(
            len(dataset), batch_size, args.seed))
        iterator = iter(loader)

    def step():
        nonlocal iterator
        if resident is not None:
            a, b = resident
        else:
            try:
                a, b = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                a, b = next(iterator)
            a, b = a.cuda(), b.cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = clip_loss(*model(a, b), args.temperature)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite benchmark loss")
        loss.backward()
        optimizer.step()
        return len(a), loss.item()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        step()
    rates, elapsed = [], []
    last_loss = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        count = 0
        for _ in range(args.steps):
            size, last_loss = step()
            count += size
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        rates.append(count / seconds)
        elapsed.append(seconds)
    return dict(mode=mode, batch_size=batch_size, status="ok",
        median_samples_per_second=statistics.median(rates),
        samples_per_second_repeats=rates, elapsed_seconds_repeats=elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        parameter_count=sum(p.numel() for p in model.parameters()), last_synthetic_loss=last_loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--modes", nargs="+", choices=["resident", "loader"], default=["resident", "loader"])
    parser.add_argument("--input-dim-a", type=int, default=1280)
    parser.add_argument("--input-dim-b", type=int, default=1600)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batch_sizes) < 2 or max(args.batch_sizes) > args.samples:
        raise ValueError("Batch sizes must be between 2 and samples")
    if min(args.steps, args.repeats, args.warmup, args.threads) < 1:
        raise ValueError("Timing settings must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
        cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                            "--format=csv,noheader"], text=True).strip(),
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        precision="FP32; TF32 disabled; no AMP; deterministic algorithms",
        source_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1] / "src/pepclip" / name).read_bytes()).hexdigest()
                       for name in ("model.py", "cache.py", "train.py")},
        limitations=["Synthetic input dimensions, not measured encoder outputs",
            "No pretrained encoder execution or real-data accuracy evaluation",
            "Loader uses a warm OS page-cache synthetic dataset; not a cold full-scale SSD test",
            "Loader matches current num_workers=0, unpinned, blocking transfer path",
            "Resident reuses one GPU-resident input batch; excludes data loading and transfer",
            "Peak allocator memory includes warmup and training, but excludes CUDA driver/context allocations"],
        cases=[])
    write_json(args.output, result)
    dataset = make_dataset(args) if "loader" in args.modes else None
    for mode in args.modes:
        for batch in args.batch_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                case = run_case(args, batch, mode, dataset)
            except torch.OutOfMemoryError:
                case = dict(mode=mode, batch_size=batch, status="out_of_memory")
            result["cases"].append(case)
            write_json(args.output, result)
            print(json.dumps(case), flush=True)
    result.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output, result)


if __name__ == "__main__":
    main()
