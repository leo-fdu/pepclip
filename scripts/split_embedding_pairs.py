"""Create a reproducible train/val manifest from matching extracted bundles.

No features are copied. SQLite bounds exact ID/sequence uniqueness checks.
"""
import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sqlite3

from pepclip.data import AMINO_ACIDS, read_fasta
from pepclip.disk_data import iter_ids
from pepclip.utils import file_sha256, fingerprint, write_json


def split_pairs(fasta, cache_a, cache_b, output, seed=42, val_fraction=0.15):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    fasta, cache_a, cache_b, output = map(Path, (fasta, cache_a, cache_b, output))
    if output.exists():
        raise FileExistsError(output)
    metas = [json.loads((p / 'metadata.json').read_text()) for p in (cache_a, cache_b)]
    source_hash = file_sha256(fasta)
    for path, meta in zip((cache_a, cache_b), metas):
        if meta['schema_version'] != 2:
            raise ValueError('Schema-2 bundles required')
        if meta['provenance']['run_identity']['source_hashes']['fasta'] != source_hash:
            raise ValueError('FASTA differs from extraction source')
        if file_sha256(path / 'ids.json') != meta['ids_sha256']:
            raise ValueError('Corrupt bundle IDs')
    if metas[0]['ids_sha256'] != metas[1]['ids_sha256'] or metas[0]['shape'][0] != metas[1]['shape'][0]:
        raise ValueError('Expected identical source-order IDs in both bundles')
    output.mkdir(parents=True)
    db_path = output / 'audit.sqlite'
    counts = Counter(train=0, val=0, test=0)
    with sqlite3.connect(db_path) as db, (output / 'pairs.jsonl.tmp').open('w') as dest:
        db.execute('PRAGMA cache_size=-8192')
        db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE seen (sid TEXT PRIMARY KEY, sequence TEXT UNIQUE)')
        for n, (record, sid) in enumerate(zip_longest(read_fasta(fasta), iter_ids(cache_a / 'ids.json')), 1):
            if record is None or sid is None or record[0].split()[0] != sid:
                raise ValueError('FASTA and embedding ID order/count differ')
            sequence = record[1]
            if not 10 <= len(sequence) <= 100 or set(sequence) - AMINO_ACIDS:
                raise ValueError('Invalid peptide sequence')
            try:
                db.execute('INSERT INTO seen VALUES (?,?)', (sid, sequence))
            except sqlite3.IntegrityError as error:
                raise ValueError(f'Duplicate ID or exact sequence at {sid}') from error
            value = int(fingerprint([seed, sid])[:16], 16) / 2**64
            split = 'val' if value < val_fraction else 'train'
            row = dict(sample_id=sid, split=split,
                       identity_id=hashlib.sha256(sequence.encode()).hexdigest(), group_id=sid)
            dest.write(json.dumps(row, separators=(',', ':')) + '\n')
            counts[split] += 1
            if n % 100000 == 0:
                db.commit()
                print(json.dumps(dict(processed=n, counts=counts)), flush=True)
        if sum(counts.values()) != metas[0]['shape'][0] or min(counts['train'], counts['val']) < 2:
            raise ValueError('Unexpected count or insufficient train/val samples')
        db.commit()
    db_path.unlink()
    (output / 'pairs.jsonl.tmp').replace(output / 'pairs.jsonl')
    result = dict(status='complete', seed=seed, train_fraction=1-val_fraction,
                  val_fraction=val_fraction, test_fraction=0, counts=dict(counts),
                  split_policy='SHA256 JSON [seed, sample_id]; val below threshold, otherwise train; approximate proportions',
                  identity_policy='exact sequence uniqueness audited in SQLite; identity_id is sequence SHA256',
                  similarity_policy='no homology clustering; not a strict low-homology split',
                  fasta_sha256=source_hash, ids_sha256=metas[0]['ids_sha256'],
                  pairs_sha256=file_sha256(output / 'pairs.jsonl'),
                  features_sha256=[m['features_sha256'] for m in metas],
                  features_validation='metadata hashes recorded; feature bytes not rehashed by this split command')
    write_json(output / 'metadata.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fasta', 'embedding-a', 'embedding-b', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(split_pairs(args.fasta, args.embedding_a, args.embedding_b,
                                 args.output, args.seed, args.val_fraction), indent=2))


if __name__ == '__main__':
    main()
