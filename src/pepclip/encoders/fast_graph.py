"""Lossless GROVER acceleration for one explicitly pinned upstream source.

Only skips atom pairs with no bond. Bond order, features, and atom numbering
remain exactly those of the original MolGraph. Never enabled implicitly.
"""
import inspect
import textwrap

from ..utils import file_sha256

UPSTREAM_SHA256 = "0e87f105b8635cff23594e1a861640ead52d161aac4ef537853340500d9c8eb7"


def fast_molgraph_class():
    import grover.data.molgraph as upstream
    if file_sha256(upstream.__file__) != UPSTREAM_SHA256:
        raise ValueError("Fast graph builder requires the validated GROVER source hash")
    source = textwrap.dedent(inspect.getsource(upstream.MolGraph))
    old = '''        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue
'''
    new = '''        edges = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                        max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                       for b in mol.GetBonds())
        for a1, a2 in edges:
                bond = mol.GetBondBetweenAtoms(a1, a2)
'''
    if source.count(old) != 1:
        raise ValueError("Unexpected upstream bond traversal")
    namespace = dict(vars(upstream))
    exec(compile(source.replace(old, new), "<pepclip-fast-molgraph>", "exec"), namespace)
    return namespace["MolGraph"]


def pack_graph(components):
    """Compact float32/int64 components without changing any restored bits."""
    import io
    import numpy as np
    arrays = {f"c{i}": value.numpy() for i, value in enumerate(components)}
    atom_dim = arrays["c0"].shape[1]
    if not np.array_equal(arrays["c1"][:, :atom_dim], arrays["c0"][arrays["c3"]]):
        raise ValueError("Unexpected bond feature prefix")
    if not np.array_equal(arrays["c7"], arrays["c3"][arrays["c2"]]):
        raise ValueError("Unexpected atom neighbors")
    packed = {"schema": np.array(1, dtype=np.uint8)}
    for prefix, values in (("atoms", arrays["c0"]), ("bonds", arrays["c1"][:, atom_dim:])):
        binary = np.all((values == 0) | (values == 1), axis=0)
        packed[prefix + "_columns"] = binary
        packed[prefix + "_bits"] = np.packbits(values[:, binary].astype(np.uint8), axis=1)
        packed[prefix + "_other"] = values[:, ~binary]
    for key in ("c2", "c3", "c4", "c5", "c6"):
        value = arrays[key]
        if value.dtype != np.int64 or value.min() < 0:
            raise ValueError("Expected nonnegative int64 graph indices")
        narrowed = value.astype(np.uint16 if value.max() <= 65535 else np.uint32)
        if not np.array_equal(narrowed.astype(np.int64), value):
            raise ValueError("Graph index overflow")
        packed[key] = narrowed
    stream = io.BytesIO()
    np.savez_compressed(stream, **packed)
    return stream.getvalue()


def unpack_graph(blob):
    import io
    import numpy as np
    import torch
    with np.load(io.BytesIO(blob), allow_pickle=False) as packed:
        if packed["schema"].item() != 1:
            raise ValueError("Unsupported packed graph schema")
        arrays = {key: packed[key].astype(np.int64) for key in ("c2", "c3", "c4", "c5", "c6")}
        features = {}
        for prefix in ("atoms", "bonds"):
            binary, other = packed[prefix + "_columns"], packed[prefix + "_other"]
            values = np.empty((len(other), len(binary)), dtype=other.dtype)
            values[:, binary] = np.unpackbits(packed[prefix + "_bits"], axis=1, count=int(binary.sum()))
            values[:, ~binary] = other
            features[prefix] = values
        arrays["c0"] = features["atoms"]
        arrays["c1"] = np.concatenate((arrays["c0"][arrays["c3"]], features["bonds"]), axis=1)
        arrays["c7"] = arrays["c3"][arrays["c2"]]
    return tuple(torch.from_numpy(arrays[f"c{i}"]) for i in range(8))
