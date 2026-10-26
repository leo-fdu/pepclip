"""Sequence-only UniRef90 filtering using the Python standard library."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .data import AMINO_ACIDS, read_fasta
from .utils import write_json


def filter_fasta(fasta, output, source_version, progress_every=1_000_000):
    """Keep complete canonical 10–100 aa sequences; defer graphs and splits."""
    if not source_version.strip() or progress_every < 1:
        raise ValueError("Provide a source version (or explicit unknown) and positive progress interval")
    source = Path(fasta).resolve()
    initial_stat = source.stat()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    counts, lengths, invalid = Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    database = output / "dedup.sqlite3"
    # Disk-backed exact deduplication bounds memory even for the full release.
    with sqlite3.connect(database) as seen, \
            (output / "peptides.fasta").open("w") as dest, \
            (output / "rejected.jsonl").open("w") as rejected:
        seen.execute("PRAGMA cache_size=-65536")
        seen.execute("CREATE TABLE sequences (sequence TEXT PRIMARY KEY, sample_id TEXT UNIQUE) WITHOUT ROWID")
        for header, sequence in read_fasta(source):
            digest.update((header + "\n" + sequence + "\n").encode())
            counts["total"] += 1
            sid = header.split()[0]
            if not sid.startswith("UniRef90_"):
                raise ValueError(f"Expected UniRef90 representative ID, got {sid}")
            if not 10 <= len(sequence) <= 100:
                counts["length"] += 1
            elif set(sequence) - AMINO_ACIDS:
                counts["nonstandard_residue"] += 1
                invalid.update(set(sequence) - AMINO_ACIDS)
                rejected.write(json.dumps(dict(sample_id=sid, reason="nonstandard_residue")) + "\n")
            else:
                previous = seen.execute("SELECT sample_id FROM sequences WHERE sequence=?", (sequence,)).fetchone()
                if previous:
                    counts["duplicate_sequence"] += 1
                    rejected.write(json.dumps(dict(sample_id=sid, reason="duplicate_sequence",
                                                   duplicate_of=previous[0])) + "\n")
                else:
                    seen.execute("INSERT INTO sequences VALUES (?, ?)", (sequence, sid))
                    dest.write(f">{header}\n{sequence}\n")
                    counts["retained"] += 1
                    lengths[len(sequence)] += 1
            if counts["total"] % progress_every == 0:
                seen.commit()
                progress = dict(counts=counts, elapsed_seconds=round(time.monotonic() - timer, 1))
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        seen.commit()
    final_stat = source.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError("Source file changed during parsing; outputs are incomplete")
    metadata = dict(status="complete", source="UniRef90 representatives", source_version=source_version,
        source_path=str(source), source_size_bytes=initial_stat.st_size,
        source_mtime_ns=initial_stat.st_mtime_ns, record_sha256=digest.hexdigest(),
        record_hash_policy="sha256(header + newline + concatenated_sequence + newline), UTF-8, all input records",
        started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=round(time.monotonic() - timer, 1), counts=dict(counts),
        retained_length_histogram=dict(sorted(lengths.items())),
        invalid_residue_record_counts=dict(sorted(invalid.items())),
        sequence_policy="10–100 characters inclusive; uppercase ACDEFGHIKLMNPQRSTVWY only; no cropping or replacement",
        duplicate_policy="exact sequence; retain first source header; log duplicate-to-retained mapping",
        rejection_order=["length", "nonstandard_residue", "duplicate_sequence"],
        length_rejections="counted only", splits="not assigned", graphs="not constructed")
    database.unlink()
    write_json(output / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-version", required=True, help="Release identifier, or explicitly unknown")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    args = parser.parse_args()
    print(json.dumps(filter_fasta(args.fasta, args.output, args.source_version, args.progress_every), indent=2))


if __name__ == "__main__":
    main()
