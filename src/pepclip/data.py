"""Streaming UniRef90 ingestion; sequence-derived graphs are standardized forms."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from .utils import fingerprint, write_json

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
GRAPH_POLICY = "rdkit-L-linear-free-amine-free-carboxylic-acid-no-disulfides-v1"


def read_fasta(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        header, pieces = None, []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(pieces)
                header, pieces = line[1:], []
                if not header:
                    raise ValueError("Empty FASTA header")
            else:
                if header is None:
                    raise ValueError("Sequence before FASTA header")
                pieces.append(line)
        if header is not None:
            yield header, "".join(pieces)


def sequence_to_smiles(sequence):
    from rdkit import Chem
    if not sequence or set(sequence) - AMINO_ACIDS:
        raise ValueError("Only uppercase standard amino acids are supported")
    mol = Chem.MolFromSequence(sequence, sanitize=True, flavor=0)
    if mol is None:
        raise ValueError("RDKit could not construct peptide")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def prepare(fasta, output, source_version, seed=42, val_fraction=0.1,
            test_fraction=0.1, groups=None):
    """Hash splitting is stable under input reordering; optional external groups."""
    if not source_version.strip():
        raise ValueError("source_version must identify the downloaded release")
    if min(val_fraction, test_fraction) < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid split fractions")
    group_map = {}
    if groups:
        with open(groups) as handle:
            for row in csv.DictReader(handle):
                sid, group = row["sample_id"], row["group_id"]
                if not group or sid in group_map:
                    raise ValueError("Empty or duplicate group mapping")
                group_map[sid] = group
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts, seen, ids = Counter(), {}, set()
    source_digest = hashlib.sha256()
    with (output / "manifest.jsonl").open("w") as dest, \
            (output / "rejected.jsonl").open("w") as rejected:
        for header, sequence in read_fasta(fasta):
            source_digest.update((header + "\n" + sequence + "\n").encode())
            counts["total"] += 1
            sid = header.split()[0]
            reason = None
            if not sid.startswith("UniRef90_"):
                raise ValueError(f"Expected UniRef90 representative ID, got {sid}")
            if not 10 <= len(sequence) <= 100:
                reason = "length"
            elif set(sequence) - AMINO_ACIDS:
                reason = "nonstandard_residue"
            elif sequence in seen:
                reason = "duplicate_sequence"
            if reason:
                counts[reason] += 1
                # Most full UniRef entries are out of range. Count them without
                # creating a rejection log proportional to the entire database.
                if reason != "length":
                    rejected.write(json.dumps({"sample_id": sid, "reason": reason,
                        "duplicate_of": seen.get(sequence)}) + "\n")
                continue
            if sid in ids:
                raise ValueError(f"Duplicate retained FASTA ID: {sid}")
            ids.add(sid)
            group = group_map[sid] if groups else sid
            value = int(fingerprint([seed, group])[:16], 16) / 2**64
            split = "test" if value < test_fraction else (
                "val" if value < test_fraction + val_fraction else "train")
            row = dict(sample_id=sid, sequence=sequence, smiles=sequence_to_smiles(sequence),
                       length=len(sequence), group_id=group, split=split,
                       source_header=header, source_version=source_version, graph_policy=GRAPH_POLICY)
            dest.write(json.dumps(row) + "\n")
            seen[sequence] = sid
            counts[split] += 1
    from rdkit import rdBase
    write_json(output / "metadata.json", dict(source="UniRef90 representatives",
        source_version=source_version, record_sha256=source_digest.hexdigest(),
        seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
        split_policy="external_groups" if groups else "uniref90_representative_hash",
        graph_policy=GRAPH_POLICY, rdkit_version=rdBase.rdkitVersion, counts=dict(counts)))
    return dict(counts)


def load_manifest(path):
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    ids, sequences, group_splits = set(), set(), {}
    for row in rows:
        sid, seq = row["sample_id"], row["sequence"]
        if sid in ids or seq in sequences:
            raise ValueError("Duplicate sample ID or sequence in manifest")
        if not 10 <= len(seq) <= 100 or set(seq) - AMINO_ACIDS:
            raise ValueError(f"Invalid sequence: {sid}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid split")
        if not row["smiles"] or row["graph_policy"] != GRAPH_POLICY:
            raise ValueError("Invalid graph policy or empty SMILES")
        group = row["group_id"]
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError("Group crosses data splits")
        group_splits[group] = row["split"]
        ids.add(sid)
        sequences.add(seq)
    return rows


def manifest_fingerprint(rows):
    return fingerprint(sorted(rows, key=lambda row: row["sample_id"]))


def load_pairs(path):
    """One positive pair per unique ID; each side must map the ID to that sample."""
    rows, ids, identities, group_splits = [], set(), set(), {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid, split = row.get("sample_id"), row.get("split")
            if not isinstance(sid, str) or not sid.strip() or sid in ids:
                raise ValueError("Pair manifest has empty or duplicate sample ID")
            if split not in {"train", "val", "test"}:
                raise ValueError("Pair manifest split must be train, val, or test")
            # Optional molecular/sequence identity keys catch duplicates across IDs.
            identity = row.get("identity_id", sid)
            group = row.get("group_id", sid)
            if any(not isinstance(value, str) or not value.strip() for value in (identity, group)):
                raise ValueError("identity_id and group_id must be nonempty strings")
            if identity in identities:
                raise ValueError("Duplicate identity_id: deduplicate positives before training")
            if group in group_splits and group_splits[group] != split:
                raise ValueError("Similarity group crosses data splits")
            ids.add(sid)
            identities.add(identity)
            group_splits[group] = split
            # Only pairing and split semantics enter this manifest fingerprint.
            rows.append(dict(sample_id=sid, split=split, identity_id=identity, group_id=group))
    if not rows:
        raise ValueError("Empty pair manifest")
    # Stable order fixes validation batch membership across input-line reorderings.
    return sorted(rows, key=lambda row: row["sample_id"])
