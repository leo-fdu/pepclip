from dataclasses import asdict, dataclass
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import time
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
    hidden_dim_a: int = 0  # 0 inherits legacy hidden_dim
    hidden_dim_b: int = 0
    low_memory: int = 0  # schema-2 disk index and bounded shuffle
    index_dir: str = ""  # temporary disk index parent; empty uses system temp
    shuffle_buffer: int = 65536
    projection_dim: int = 256
    temperature: float = 0.07
    learnable_temperature: int = 0
    save_every: int = 0  # optional weights-only epoch snapshots
    batch_size: int = 256
    validation_batch_size: int = 256
    validation_mode: str = "batch"
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "cpu"

    def validate(self):
        if type(self.learnable_temperature) is not int or self.learnable_temperature not in (0, 1):
            raise ValueError("Invalid learnable_temperature")
        if self.learnable_temperature and not 0.01 <= self.temperature <= 1.0:
            raise ValueError("Learnable initial temperature must be in [0.01, 1]")
        if type(self.save_every) is not int or self.save_every < 0:
            raise ValueError("Invalid save_every")
        for key in ("hidden_dim_a", "hidden_dim_b"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if (type(self.low_memory) is not int or self.low_memory not in (0, 1) or
                type(self.shuffle_buffer) is not int or self.shuffle_buffer < 2):
            raise ValueError("Invalid low_memory/shuffle_buffer")
        if self.low_memory and self.validation_mode != "batch":
            raise ValueError("low_memory requires batch validation")
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
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed))
        start = 0
        while start < self.size:
            end = min(start + self.batch_size, self.size)
            if self.size - end == 1:
                end = self.size
            yield order[start:end].tolist()
            start = end

    def __len__(self):
        n = math.ceil(self.size / self.batch_size)
        return n - int(n > 1 and self.size % self.batch_size == 1)


class BufferedBatches(ContrastiveBatches):
    """Bounded replacement-buffer shuffle, not a uniform global permutation."""
    def __init__(self, size, batch_size, seed, buffer_size=65536):
        super().__init__(size, batch_size, seed)
        self.buffer_size = buffer_size

    def __iter__(self):
        rng = random.Random(self.seed)
        buffer = list(range(min(self.size, self.buffer_size)))
        batch = []
        def indices():
            for i in range(len(buffer), self.size):
                j = rng.randrange(len(buffer))
                yield buffer[j]
                buffer[j] = i
            rng.shuffle(buffer)
            yield from buffer
        # Hold one pending batch so a trailing singleton can be merged.
        pending = None
        for i in indices():
            batch.append(i)
            if len(batch) == self.batch_size:
                if pending is not None:
                    yield pending
                pending, batch = batch, []
        if pending is None:
            yield batch
        elif len(batch) == 1:
            yield pending + batch
        else:
            yield pending
            if batch:
                yield batch


def load_inputs(manifest, cache_a, cache_b, config):
    schema_a = json.loads((Path(cache_a) / "metadata.json").read_text())["schema_version"]
    schema_b = json.loads((Path(cache_b) / "metadata.json").read_text())["schema_version"]
    if schema_a == schema_b == 2:
        if config.protein_encoder or config.graph_encoder:
            raise ValueError("Schema-2 bundles declare their provenance; encoder-name flags are legacy-only")
        from .embeddings import EmbeddingCache
        if config.low_memory:
            from .disk_data import disk_inputs
            return disk_inputs(manifest, cache_a, cache_b, config.index_dir)
        rows = load_pairs(manifest)
        a, b = EmbeddingCache(cache_a, rows), EmbeddingCache(cache_b, rows)
        provenance = dict(pairs_sha256=manifest_fingerprint(rows), a=a.metadata, b=b.metadata)
        return rows, a, b, provenance
    if schema_a != 1 or schema_b != 1:
        raise ValueError("Use two schema-2 bundles, or two legacy schema-1 peptide caches")
    if config.low_memory:
        raise ValueError("low_memory requires schema-2 embedding bundles")
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
    print(json.dumps(dict(event="loading_inputs", low_memory=config.low_memory)), flush=True)
    rows, a, b, provenance = load_inputs(manifest, cache_a, cache_b, config)
    training = PairedFeatures(rows, a, b, "train")
    validation = PairedFeatures(rows, a, b, "val")
    print(json.dumps(dict(event="inputs_ready", train_samples=len(training),
                          val_samples=len(validation))), flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    args = dict(input_dim_a=a.output_dim, input_dim_b=b.output_dim,
                hidden_dim=config.hidden_dim, projection_dim=config.projection_dim,
                hidden_dim_a=config.hidden_dim_a, hidden_dim_b=config.hidden_dim_b,
                learnable_temperature=config.learnable_temperature, temperature=config.temperature)
    model = EmbeddingCLIP(**args).to(config.device)
    parameters = model.parameters()
    if config.learnable_temperature:
        parameters = [dict(params=[p for n, p in model.named_parameters() if n != "logit_scale"]),
                      dict(params=[model.logit_scale], weight_decay=0.0)]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    start, best = 0, math.inf
    if resume:
        checkpoint = load_checkpoint(resume)
        previous_config = asdict(TrainConfig(**checkpoint["config"]))
        if "validation_mode" not in checkpoint["config"]:
            # Old checkpoints selected best.pt using full-candidate retrieval loss.
            previous_config.update(validation_mode="retrieval",
                                   validation_batch_size=previous_config["batch_size"])
        for key, value in asdict(config).items():
            if key not in {"epochs", "device", "index_dir", "save_every"} and previous_config[key] != value:
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
        del checkpoint  # release CPU copies of restored model/optimizer tensors
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
        epoch_started = time.monotonic()
        sampler = BufferedBatches if config.low_memory else ContrastiveBatches
        extra = dict(buffer_size=config.shuffle_buffer) if config.low_memory else {}
        loader = DataLoader(training, batch_sampler=sampler(
            len(training), config.batch_size, config.seed + epoch, **extra), num_workers=0)
        for step, (a, b) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            z_a, z_b = model(a.to(config.device), b.to(config.device))
            loss = clip_loss(z_a, z_b, model.loss_temperature(config.temperature))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            model.clamp_temperature()
            total += loss.item() * len(a)
            count += len(a)
            if step == 1 or step % 1000 == 0:
                print(json.dumps(dict(event="training_progress", epoch=epoch+1, step=step,
                    samples=count, train_samples=len(training), mean_loss=total/count,
                    elapsed_seconds=round(time.monotonic()-epoch_started, 1))), flush=True)
        metrics = evaluate(model, validation, config.device, config.validation_batch_size,
                           config.temperature, config.validation_mode)
        if not math.isfinite(metrics["loss"]):
            raise ValueError("Nonfinite validation loss; checkpoint was not saved")
        if metrics["loss"] < best:
            best = metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        record = dict(epoch=epoch+1, train_loss=total/count, validation=metrics,
                      temperature=float(torch.as_tensor(model.loss_temperature(config.temperature)).detach()))
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
        if config.save_every and (epoch + 1) % config.save_every == 0:
            filename = output / f"epoch_{epoch + 1:04d}.pt"
            temporary = filename.with_suffix(".pt.tmp")
            torch.save(dict(schema_version=2, model_args=args, config=asdict(config),
                            provenance=provenance, epoch=epoch+1, model=model.state_dict()), temporary)
            temporary.replace(filename)
    return output / "last.pt"
