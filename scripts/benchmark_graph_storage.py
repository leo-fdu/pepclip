"""Experimental lossless packing of existing GROVER CPU benchmark shards.

This measures storage only; the production encoder reader does not use this codec.
Keep the original index.json alongside packed shards to preserve sample identities.
"""
import argparse
import io
import json
from pathlib import Path
import time

import numpy as np


def pack_features(arrays, prefix, features):
    binary = np.all((features == 0) | (features == 1), axis=0)
    arrays[prefix + "_binary_columns"] = binary
    arrays[prefix + "_binary_bits"] = np.packbits(features[:, binary].astype(np.uint8), axis=1)
    arrays[prefix + "_other"] = features[:, ~binary]


def unpack_features(arrays, prefix):
    binary = arrays[prefix + "_binary_columns"]
    other = arrays[prefix + "_other"]
    restored = np.empty((len(other), len(binary)), dtype=other.dtype)
    restored[:, binary] = np.unpackbits(arrays[prefix + "_binary_bits"], axis=1,
                                      count=int(binary.sum()))
    restored[:, ~binary] = other
    return restored


def encode(source):
    atom_dim = source["c0"].shape[1]
    if not np.array_equal(source["c1"][:, :atom_dim], source["c0"][source["c3"]]):
        raise ValueError("Bond atom-prefix relation does not hold")
    if not np.array_equal(source["c7"], source["c3"][source["c2"]]):
        raise ValueError("Neighbor relation does not hold")
    packed = {"schema_version": np.array(1, dtype=np.uint8)}
    pack_features(packed, "atoms", source["c0"])
    pack_features(packed, "bonds", source["c1"][:, atom_dim:])
    for key in ("c2", "c3", "c4", "c5", "c6"):
        value = source[key]
        if value.dtype != np.int64 or value.min() < 0:
            raise ValueError("Expected nonnegative int64 indices")
        dtype = np.uint16 if value.max() <= 65535 else np.uint32
        narrowed = value.astype(dtype)
        if not np.array_equal(narrowed.astype(np.int64), value):
            raise ValueError("Index narrowing would lose information")
        packed[key] = narrowed
    stream = io.BytesIO()
    np.savez_compressed(stream, **packed)
    return stream.getvalue()


def decode(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as packed:
        if packed["schema_version"].item() != 1:
            raise ValueError("Unsupported schema")
        restored = {key: packed[key].astype(np.int64) for key in ("c2", "c3", "c4", "c5", "c6")}
        restored["c0"] = unpack_features(packed, "atoms")
        restored["c1"] = np.concatenate((restored["c0"][restored["c3"]],
                                          unpack_features(packed, "bonds")), axis=1)
        restored["c7"] = restored["c3"][restored["c2"]]
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunks < 1 or args.output.exists():
        raise ValueError("Require positive chunk count and new output file")
    index = json.loads((args.input / "index.json").read_text())[:args.chunks]
    results = []
    for record in index:
        path = args.input / f"chunk-{record['chunk']:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            source = dict(archive)
        start = time.perf_counter()
        blob = encode(source)
        after_encode = time.perf_counter()
        restored = decode(blob)
        after_decode = time.perf_counter()
        for key, value in source.items():
            if (value.dtype != restored[key].dtype or value.shape != restored[key].shape
                    or value.tobytes() != restored[key].tobytes()):
                raise AssertionError(f"Bitwise roundtrip failed for {key}")
        results.append(dict(chunk=record["chunk"], samples=record["samples"],
                            original_bytes=path.stat().st_size, packed_bytes=len(blob),
                            encode_seconds=after_encode-start, decode_seconds=after_decode-after_encode))
    summary = {key: sum(row[key] for row in results) for key in
               ("samples", "original_bytes", "packed_bytes", "encode_seconds", "decode_seconds")}
    summary.update(bitwise_roundtrip=True, chunks=len(results), numpy_version=np.__version__,
                   input=str(args.input), cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
