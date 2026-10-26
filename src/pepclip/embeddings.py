"""Encoder-independent, immutable embedding bundles and explicit sample splits."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .utils import file_sha256, write_json


def _read_ids(path):
    ids = json.loads(Path(path).read_text())
    if not isinstance(ids, list) or not ids or any(not isinstance(s, str) or not s.strip() for s in ids):
        raise ValueError("ids.json must be a nonempty array of nonempty sample ID strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate embedding sample IDs")
    return ids


def _validate_array(path, count):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 1:
        raise ValueError("Embeddings must have shape [number of IDs, positive feature dimension]")
    if array.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Embeddings must be float16, float32, or float64")
    for start in range(0, count, 8192):
        # Training is FP32; also reject finite float64 values that overflow FP32.
        with np.errstate(over="ignore", invalid="ignore"):
            chunk = np.asarray(array[start:start + 8192], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError("Embeddings contain nonfinite or FP32-unrepresentable values")
    return array


def _validate_provenance(provenance):
    required = ("encoder", "version", "pooling", "preprocessing")
    if not isinstance(provenance, dict) or any(
            not isinstance(provenance.get(key), str) or not provenance[key].strip() for key in required):
        raise ValueError("Provenance requires encoder, version, pooling, preprocessing strings; use explicit unknowns")


def import_embeddings(features, ids, provenance, output):
    """Copy externally computed vectors into a portable, checksummed bundle.

    No encoder is loaded. Existing output directories are never overwritten.
    metadata.json is written last and marks successful completion.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    provenance = json.loads(Path(provenance).read_text())
    _validate_provenance(provenance)
    sample_ids = _read_ids(ids)
    original_hash = file_sha256(features)
    array = _validate_array(features, len(sample_ids))
    shape, dtype = list(array.shape), str(array.dtype)
    del array
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Hash before copying, then verify the copy to detect source changes/copy errors.
    shutil.copyfile(features, output / "features.npy")
    if file_sha256(output / "features.npy") != original_hash:
        raise ValueError("Embedding source changed during import or copy failed")
    write_json(output / "ids.json", sample_ids)
    metadata = dict(schema_version=2, shape=shape, dtype=dtype, provenance=provenance,
        features_sha256=original_hash, ids_sha256=file_sha256(output / "ids.json"))
    write_json(output / "metadata.json", metadata)
    return metadata


class EmbeddingCache:
    """An immutable bundle reusable across subsets, splits, and projection heads."""
    def __init__(self, path, rows):
        path = Path(path)
        self.metadata = json.loads((path / "metadata.json").read_text())
        meta = self.metadata
        if meta.get("schema_version") != 2:
            raise ValueError("Expected an encoder-independent schema-2 embedding bundle")
        _validate_provenance(meta.get("provenance"))
        for name, filename in (("features", "features.npy"), ("ids", "ids.json")):
            if file_sha256(path / filename) != meta[name + "_sha256"]:
                raise ValueError(f"Corrupt {name} cache")
        ids = _read_ids(path / "ids.json")
        self.indices = {sid: i for i, sid in enumerate(ids)}
        missing = next((row["sample_id"] for row in rows if row["sample_id"] not in self.indices), None)
        if missing is not None:
            raise ValueError(f"Embedding cache is missing selected sample: {missing}")
        self.features = _validate_array(path / "features.npy", len(ids))
        if list(self.features.shape) != meta["shape"] or str(self.features.dtype) != meta["dtype"]:
            raise ValueError("Embedding shape/dtype differs from metadata")
        self.output_dim = self.features.shape[1]

    def get(self, sid):
        return torch.from_numpy(np.array(self.features[self.indices[sid]], dtype=np.float32, copy=True))
