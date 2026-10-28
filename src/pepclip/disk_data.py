"""Schema-2 training inputs with disk-backed metadata and bounded feature reads.

Only stdlib SQLite is required. Temporary indexes are rebuilt and validated on
startup; source bundles and split assignments are never modified.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import torch

from .embeddings import _validate_provenance
from .utils import file_sha256


def iter_ids(path):
    """Stream the existing JSON string array, including pretty-printed arrays."""
    decoder = json.JSONDecoder()
    with open(path) as handle:
        buffer, eof = "", False

        def fill():
            nonlocal buffer, eof
            chunk = handle.read(65536)
            buffer += chunk
            eof = not chunk

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError("ids.json must be an array")
        buffer = buffer[1:]
        first = True
        while True:
            whitespace()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Invalid IDs JSON separator")
                buffer = buffer[1:]
                whitespace()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof or len(buffer) > 1024 * 1024:
                        raise ValueError("Invalid or oversized sample ID")
                    fill()
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Empty or non-string sample ID")
            yield value
            buffer = buffer[end:]
            first = False
        whitespace()
        if buffer or not eof:
            raise ValueError("Trailing IDs JSON data")
        if first:
            raise ValueError("Empty IDs")


class DiskFeatures:
    def __init__(self, path, db, table):
        path = Path(path)
        self.metadata = json.loads((path / "metadata.json").read_text())
        meta = self.metadata
        if meta.get("schema_version") != 2:
            raise ValueError("Disk inputs require schema-2 bundles")
        _validate_provenance(meta.get("provenance"))
        for name, filename in (("features", "features.npy"), ("ids", "ids.json")):
            if file_sha256(path / filename) != meta[name + "_sha256"]:
                raise ValueError(f"Corrupt {name} cache")
        db.execute(f"CREATE TABLE {table} (sid TEXT PRIMARY KEY, pos INTEGER NOT NULL)")
        count = 0
        try:
            for count, sid in enumerate(iter_ids(path / "ids.json"), 1):
                db.execute(f"INSERT INTO {table} VALUES (?, ?)", (sid, count - 1))
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate embedding sample IDs") from error
        array = np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
        if (array.ndim != 2 or array.shape[0] != count or array.shape[1] < 1 or
                array.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")) or
                list(array.shape) != meta["shape"] or str(array.dtype) != meta["dtype"]):
            raise ValueError("Invalid embedding shape/dtype")
        # Row reads below require C order; do not silently reinterpret Fortran arrays.
        if not array.flags.c_contiguous:
            raise ValueError("low_memory requires C-order features.npy; re-export in C order")
        self.offset, self.dtype = array.offset, array.dtype
        self.output_dim = array.shape[1]
        self.row_bytes = self.output_dim * self.dtype.itemsize
        del array
        self.handle = open(path / "features.npy", "rb")
        self.handle.seek(self.offset)
        remaining = count
        chunk_rows = max(1, (4 * 1024 * 1024) // self.row_bytes)
        while remaining:
            n = min(remaining, chunk_rows)
            values = np.frombuffer(self.handle.read(n * self.row_bytes), dtype=self.dtype)
            with np.errstate(over="ignore", invalid="ignore"):
                valid = np.isfinite(values.astype(np.float32)).all()
            if values.size != n * self.output_dim or not valid:
                raise ValueError("Truncated or nonfinite features")
            remaining -= n

    def get_position(self, index):
        self.handle.seek(self.offset + index * self.row_bytes)
        data = self.handle.read(self.row_bytes)
        if len(data) != self.row_bytes:
            raise ValueError("Truncated feature row")
        return torch.from_numpy(np.frombuffer(data, dtype=self.dtype).astype(np.float32, copy=True))

    def __del__(self):
        if hasattr(self, "handle"):
            self.handle.close()


class DiskPairs:
    def __init__(self, directory):
        self.temp = tempfile.TemporaryDirectory(prefix="pepclip-index-", dir=directory or None)
        self.db = sqlite3.connect(str(Path(self.temp.name) / "index.sqlite"))
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA mmap_size=0")

    def split_count(self, split):
        return self.counts.get(split, 0)

    def positions(self, split, index):
        result = self.db.execute("SELECT a,b FROM samples WHERE split=? AND ordinal=?",
                                 (split, index)).fetchone()
        if result is None:
            raise IndexError(index)
        return result

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()
        if hasattr(self, "temp"):
            self.temp.cleanup()


def disk_inputs(manifest, cache_a, cache_b, directory=""):
    rows = DiskPairs(directory)
    db = rows.db
    db.execute("CREATE TABLE pairs (sid TEXT PRIMARY KEY, split TEXT, identity TEXT UNIQUE, grp TEXT)")
    db.execute("CREATE TABLE groups (grp TEXT PRIMARY KEY, split TEXT)")
    try:
        with open(manifest) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid, split = row.get("sample_id"), row.get("split")
                identity, group = row.get("identity_id", sid), row.get("group_id", sid)
                if (any(not isinstance(v, str) or not v.strip() for v in (sid, identity, group)) or
                        split not in {"train", "val", "test"}):
                    raise ValueError("Invalid pair manifest")
                db.execute("INSERT INTO pairs VALUES (?,?,?,?)", (sid, split, identity, group))
                db.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (group, split))
                if db.execute("SELECT split FROM groups WHERE grp=?", (group,)).fetchone()[0] != split:
                    raise ValueError("Similarity group crosses data splits")
    except sqlite3.IntegrityError as error:
        raise ValueError("Duplicate sample_id or identity_id") from error
    if not db.execute("SELECT 1 FROM pairs LIMIT 1").fetchone():
        raise ValueError("Empty pair manifest")
    a, b = DiskFeatures(cache_a, db, "ids_a"), DiskFeatures(cache_b, db, "ids_b")
    missing = db.execute("SELECT p.sid FROM pairs p LEFT JOIN ids_a a ON p.sid=a.sid "
                         "LEFT JOIN ids_b b ON p.sid=b.sid WHERE a.pos IS NULL OR b.pos IS NULL LIMIT 1").fetchone()
    if missing:
        raise ValueError(f"Missing embedding sample: {missing[0]}")
    db.execute("CREATE TABLE samples (split TEXT, ordinal INTEGER, a INTEGER, b INTEGER, PRIMARY KEY(split,ordinal))")
    rows.counts = {}
    digest = hashlib.sha256(b"[")
    first = True
    # Matches manifest_fingerprint(load_pairs(...)) byte for byte without materializing rows.
    for sid, split, identity, group, ia, ib in db.execute(
            "SELECT p.sid,p.split,p.identity,p.grp,a.pos,b.pos FROM pairs p "
            "JOIN ids_a a ON p.sid=a.sid JOIN ids_b b ON p.sid=b.sid ORDER BY p.sid"):
        record = dict(sample_id=sid, split=split, identity_id=identity, group_id=group)
        digest.update(("" if first else ", ").encode())
        digest.update(json.dumps(record, sort_keys=True).encode())
        first = False
        ordinal = rows.counts.get(split, 0)
        db.execute("INSERT INTO samples VALUES (?,?,?,?)", (split, ordinal, ia, ib))
        rows.counts[split] = ordinal + 1
    digest.update(b"]")
    for table in ("pairs", "groups", "ids_a", "ids_b"):
        db.execute(f"DROP TABLE {table}")
    db.commit()
    return rows, a, b, dict(pairs_sha256=digest.hexdigest(), a=a.metadata, b=b.metadata)
