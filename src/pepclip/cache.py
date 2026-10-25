import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_manifest, manifest_fingerprint
from .utils import file_sha256, write_json


def extract(manifest, output, encoder, modality, batch_size=16):
    if modality not in {"sequence", "graph"} or batch_size < 1:
        raise ValueError("Invalid extraction modality/batch size")
    rows = load_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    field = "sequence" if modality == "sequence" else "smiles"
    array = np.lib.format.open_memmap(output / "features.npy", mode="w+",
                                    dtype="float32", shape=(len(rows), encoder.output_dim))
    # metadata.json is a completion marker; a failed extraction cannot be trained on.
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        try:
            features = encoder.encode([row[field] for row in batch])
        except Exception as error:
            raise RuntimeError(f"Extraction failed for IDs {[row['sample_id'] for row in batch]}") from error
        if tuple(features.shape) != (len(batch), encoder.output_dim):
            raise ValueError("Encoder output shape does not match declared dimensions")
        if not torch.isfinite(features).all():
            raise ValueError(f"Nonfinite features at batch starting {start}")
        array[start:start + len(batch)] = features.detach().cpu().numpy()
        print(f"extract {modality}: {start + len(batch)}/{len(rows)}", flush=True)
    array.flush()
    write_json(output / "ids.json", [row["sample_id"] for row in rows])
    write_json(output / "metadata.json", dict(schema_version=1, modality=modality,
        manifest_sha256=manifest_fingerprint(rows), encoder=encoder.metadata(),
        features_sha256=file_sha256(output / "features.npy"),
        ids_sha256=file_sha256(output / "ids.json")))


class FeatureCache:
    def __init__(self, path, rows, modality, encoder_name):
        path = Path(path)
        self.metadata = json.loads((path / "metadata.json").read_text())
        meta = self.metadata
        if meta["schema_version"] != 1 or meta["modality"] != modality:
            raise ValueError("Cache schema/modality mismatch")
        if meta["encoder"]["encoder"] != encoder_name:
            raise ValueError("Selected encoder differs from cached encoder")
        if meta["manifest_sha256"] != manifest_fingerprint(rows):
            raise ValueError("Cache does not match manifest (including split and inputs)")
        for name in ("features", "ids"):
            filename = path / ("features.npy" if name == "features" else "ids.json")
            if file_sha256(filename) != meta[name + "_sha256"]:
                raise ValueError(f"Corrupt {name} cache")
        ids = json.loads((path / "ids.json").read_text())
        if len(set(ids)) != len(ids) or set(ids) != {r["sample_id"] for r in rows}:
            raise ValueError("Cache IDs are duplicated or missing")
        self.indices = {sid: i for i, sid in enumerate(ids)}
        self.features = np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
        if self.features.dtype != np.float32 or self.features.shape != (len(ids), meta["encoder"]["output_dim"]):
            raise ValueError("Invalid cache shape/dtype")
        for start in range(0, len(ids), 8192):
            if not np.isfinite(self.features[start:start+8192]).all():
                raise ValueError("Nonfinite cached features")
        self.output_dim = self.features.shape[1]

    def get(self, sid):
        return torch.from_numpy(self.features[self.indices[sid]].copy())


class PairedFeatures(Dataset):
    def __init__(self, rows, sequence_cache, graph_cache, split):
        self.rows = [row for row in rows if row["split"] == split]
        self.sequence, self.graph = sequence_cache, graph_cache
        if len(self.rows) < 2:
            raise ValueError(f"Split {split} must contain at least 2 unique samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sid = self.rows[index]["sample_id"]
        return self.sequence.get(sid), self.graph.get(sid)
