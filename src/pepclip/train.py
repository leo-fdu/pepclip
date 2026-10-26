from dataclasses import asdict, dataclass
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .cache import FeatureCache, PairedFeatures
from .data import load_manifest, load_pairs, manifest_fingerprint
from .evaluate import evaluate
from .model import EmbeddingCLIP, PeptideCLIP, clip_loss
from .utils import write_json


@dataclass
class TrainConfig:
    # Encoder names are used only when reading legacy schema-1 peptide caches.
    protein_encoder: str = ""
    graph_encoder: str = ""
    hidden_dim: int = 512
    projection_dim: int = 256
    temperature: float = 0.07
    batch_size: int = 256
    validation_batch_size: int = 256
    validation_mode: str = "batch"
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "cpu"

    def validate(self):
        for key in ("hidden_dim", "projection_dim", "batch_size", "validation_batch_size", "epochs"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < (2 if key in {"batch_size", "validation_batch_size"} else 1):
                raise ValueError(f"Invalid {key}")
        if self.validation_mode not in {"batch", "retrieval"}:
            raise ValueError("Invalid validation_mode")
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


def load_inputs(manifest, cache_a, cache_b, config):
    schema_a = json.loads((Path(cache_a) / "metadata.json").read_text())["schema_version"]
    schema_b = json.loads((Path(cache_b) / "metadata.json").read_text())["schema_version"]
    if schema_a == schema_b == 2:
        if config.protein_encoder or config.graph_encoder:
            raise ValueError("Schema-2 bundles declare their provenance; encoder-name flags are legacy-only")
        from .embeddings import EmbeddingCache
        rows = load_pairs(manifest)
        a, b = EmbeddingCache(cache_a, rows), EmbeddingCache(cache_b, rows)
        provenance = dict(pairs_sha256=manifest_fingerprint(rows), a=a.metadata, b=b.metadata)
        return rows, a, b, provenance
    if schema_a != 1 or schema_b != 1:
        raise ValueError("Use two schema-2 bundles, or two legacy schema-1 peptide caches")
    # Legacy extraction remains available without constraining generic inputs.
    rows = load_manifest(manifest)
    name_a = config.protein_encoder or json.loads((Path(cache_a) / "metadata.json").read_text())["encoder"]["encoder"]
    name_b = config.graph_encoder or json.loads((Path(cache_b) / "metadata.json").read_text())["encoder"]["encoder"]
    seq = FeatureCache(cache_a, rows, "sequence", name_a)
    graph = FeatureCache(cache_b, rows, "graph", name_b)
    provenance = dict(manifest_sha256=manifest_fingerprint(rows), sequence=seq.metadata, graph=graph.metadata)
    return rows, seq, graph, provenance


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def model_from_checkpoint(checkpoint, device="cpu"):
    cls = EmbeddingCLIP if "input_dim_a" in checkpoint["model_args"] else PeptideCLIP
    model = cls(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def train(manifest, cache_a, cache_b, output, config, resume=None):
    config.validate()
    rows, a, b, provenance = load_inputs(manifest, cache_a, cache_b, config)
    training = PairedFeatures(rows, a, b, "train")
    validation = PairedFeatures(rows, a, b, "val")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    args = dict(input_dim_a=a.output_dim, input_dim_b=b.output_dim,
                hidden_dim=config.hidden_dim, projection_dim=config.projection_dim)
    model = EmbeddingCLIP(**args).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    start, best = 0, math.inf
    if resume:
        checkpoint = load_checkpoint(resume)
        previous_config = checkpoint["config"].copy()
        if "validation_mode" not in previous_config:
            # Old checkpoints selected best.pt using full-candidate retrieval loss.
            previous_config.update(validation_mode="retrieval",
                                   validation_batch_size=previous_config["batch_size"])
        for key, value in asdict(config).items():
            if key not in {"epochs", "device"} and previous_config[key] != value:
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
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        cuda_runtime=torch.version.cuda, numpy_version=np.__version__,
        cuda_device=torch.cuda.get_device_name(config.device) if torch.device(config.device).type == "cuda" else None,
        resume=str(resume) if resume else None))
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
        metrics = evaluate(model, validation, config.device, config.validation_batch_size,
                           config.temperature, config.validation_mode)
        if not math.isfinite(metrics["loss"]):
            raise ValueError("Nonfinite validation loss; checkpoint was not saved")
        if metrics["loss"] < best:
            best = metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        record = dict(epoch=epoch+1, train_loss=total/count, validation=metrics)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        checkpoint = dict(schema_version=2, model_args=args, config=asdict(config),
            provenance=provenance, model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch+1, best_val_loss=best, best_model=best_state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(checkpoint, output / "last.pt.tmp")
        (output / "last.pt.tmp").replace(output / "last.pt")
        torch.save(dict(schema_version=2, model_args=args, config=asdict(config),
                        provenance=provenance, model=best_state), output / "best.pt.tmp")
        (output / "best.pt.tmp").replace(output / "best.pt")
    return output / "last.pt"
