"""Schema-2 training inputs with disk-backed metadata and bounded feature reads.

Only stdlib SQLite is required. Temporary indexes are rebuilt and validated on
startup; source bundles and split assignments are never modified.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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


class BatchReader:
    """One owner thread; bounded batch buffers, no whole-feature memory mapping."""
    def __init__(self, dataset, path):
        self.split = dataset.split
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA mmap_size=0")
        self.sources = []
        try:
            for source in (dataset.cache_a, dataset.cache_b):
                self.sources.append((open(source.handle.name, "rb"), source))
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle, _ in self.sources:
            handle.close()
        self.db.close()

    def read(self, indices):
        # Chunk SQL parameters for SQLite builds with the legacy variable limit.
        positions = {}
        for start in range(0, len(indices), 500):
            chunk = indices[start:start + 500]
            marks = ",".join("?" for _ in chunk)
            positions.update((ordinal, (a, b)) for ordinal, a, b in self.db.execute(
                f"SELECT ordinal,a,b FROM samples WHERE split=? AND ordinal IN ({marks})",
                [self.split, *chunk]))
        if any(i not in positions for i in indices):
            raise IndexError("Missing batch sample")
        result = []
        for side, (handle, source) in enumerate(self.sources):
            output = np.empty((len(indices), source.output_dim), dtype=np.float32)
            ordered = sorted((positions[index][side], dest) for dest, index in enumerate(indices))
            start = 0
            while start < len(ordered):
                end = start + 1
                while end < len(ordered) and ordered[end][0] == ordered[end-1][0] + 1:
                    end += 1
                handle.seek(source.offset + ordered[start][0] * source.row_bytes)
                data = handle.read((end-start) * source.row_bytes)
                if len(data) != (end-start) * source.row_bytes:
                    raise ValueError("Truncated feature batch")
                values = np.frombuffer(data, dtype=source.dtype).reshape(end-start, source.output_dim)
                for row, (_, dest) in zip(values, ordered[start:end]):
                    output[dest] = row
                start = end
            result.append(torch.from_numpy(output))
        return tuple(result)


@contextmanager
def disk_batches(dataset, sampler, prefetch=False):
    """Preserve sampler order; at most one additional batch is being prepared.

    The reader and its SQLite connection are created, used and closed on the
    same thread. Shutdown also joins pending reads on early exit or failure.
    """
    path = dataset.disk_rows.db.execute("PRAGMA database_list").fetchone()[2]
    if not prefetch:
        reader = BatchReader(dataset, path)
        try:
            yield (reader.read(indices) for indices in sampler)
        finally:
            reader.close()
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pepclip-reader") as pool:
        reader = pool.submit(BatchReader, dataset, path).result()
        def batches():
            iterator = iter(sampler)
            indices = next(iterator, None)
            pending = pool.submit(reader.read, indices) if indices is not None else None
            while pending is not None:
                batch = pending.result()
                indices = next(iterator, None)
                pending = pool.submit(reader.read, indices) if indices is not None else None
                yield batch
        iterator = batches()
        try:
            yield iterator
        finally:
            iterator.close()
            pool.submit(reader.close).result()
