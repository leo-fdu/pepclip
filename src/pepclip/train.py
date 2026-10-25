from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .cache import FeatureCache, PairedFeatures
from .data import load_manifest, manifest_fingerprint
from .evaluate import evaluate
from .model import PeptideCLIP, clip_loss
from .utils import write_json


@dataclass
class TrainConfig:
    protein_encoder: str = "esm2"
    graph_encoder: str = "grover"
    hidden_dim: int = 512
    projection_dim: int = 256
    temperature: float = 0.07
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "cpu"

    def validate(self):
        for key in ("hidden_dim", "projection_dim", "batch_size", "epochs"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < (2 if key == "batch_size" else 1):
                raise ValueError(f"Invalid {key}")
        for key in ("temperature", "learning_rate", "weight_decay"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {key}")


class ContrastiveBatches(Sampler):
    """Keep every sample; merge a final singleton into the previous batch."""
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        if size < 2 or batch_size < 2:
            raise ValueError("Need at least two contrastive samples")

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed)).tolist()
        batches = [order[i:i+self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return iter(batches)

    def __len__(self):
        n = math.ceil(self.size / self.batch_size)
        return n - int(n > 1 and self.size % self.batch_size == 1)


def load_inputs(manifest, sequence_cache, graph_cache, config):
    rows = load_manifest(manifest)
    seq = FeatureCache(sequence_cache, rows, "sequence", config.protein_encoder)
    graph = FeatureCache(graph_cache, rows, "graph", config.graph_encoder)
    provenance = dict(manifest_sha256=manifest_fingerprint(rows), sequence=seq.metadata, graph=graph.metadata)
    return rows, seq, graph, provenance


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def model_from_checkpoint(checkpoint, device="cpu"):
    model = PeptideCLIP(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def train(manifest, sequence_cache, graph_cache, output, config, resume=None):
    config.validate()
    rows, seq, graph, provenance = load_inputs(manifest, sequence_cache, graph_cache, config)
    training = PairedFeatures(rows, seq, graph, "train")
    validation = PairedFeatures(rows, seq, graph, "val")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    args = dict(sequence_dim=seq.output_dim, graph_dim=graph.output_dim,
                hidden_dim=config.hidden_dim, projection_dim=config.projection_dim)
    model = PeptideCLIP(**args).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    start, best = 0, math.inf
    if resume:
        checkpoint = load_checkpoint(resume)
        for key, value in asdict(config).items():
            if key not in {"epochs", "device"} and checkpoint["config"][key] != value:
                raise ValueError(f"Resume config changed: {key}")
        if checkpoint["provenance"] != provenance:
            raise ValueError("Resume data/encoder provenance mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["epoch"], checkpoint["best_val_loss"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if config.device.startswith("cuda") and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        # Retain the best model even if this continuation never improves it.
        best_state = checkpoint["best_model"]
    else:
        best_state = None
    if start >= config.epochs:
        raise ValueError("epochs must exceed completed checkpoint epochs")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    write_json(output / "config.json", asdict(config))
    write_json(output / "provenance.json", dict(**provenance, git_commit=commit, git_dirty=dirty,
        torch_version=str(torch.__version__), numpy_version=np.__version__, resume=str(resume) if resume else None))
    for epoch in range(start, config.epochs):
        model.train()
        total, count = 0., 0
        loader = DataLoader(training, batch_sampler=ContrastiveBatches(
            len(training), config.batch_size, config.seed + epoch))
        for a, b in loader:
            optimizer.zero_grad(set_to_none=True)
            z_a, z_b = model(a.to(config.device), b.to(config.device))
            loss = clip_loss(z_a, z_b, config.temperature)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(a)
            count += len(a)
        metrics = evaluate(model, validation, config.device, config.batch_size, config.temperature)
        if not math.isfinite(metrics["loss"]):
            raise ValueError("Nonfinite validation loss; checkpoint was not saved")
        if metrics["loss"] < best:
            best = metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        record = dict(epoch=epoch+1, train_loss=total/count, validation=metrics)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        checkpoint = dict(schema_version=1, model_args=args, config=asdict(config),
            provenance=provenance, model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch+1, best_val_loss=best, best_model=best_state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(checkpoint, output / "last.pt.tmp")
        (output / "last.pt.tmp").replace(output / "last.pt")
        torch.save(dict(schema_version=1, model_args=args, config=asdict(config),
                        provenance=provenance, model=best_state), output / "best.pt.tmp")
        (output / "best.pt.tmp").replace(output / "best.pt")
    return output / "last.pt"
