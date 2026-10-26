import gzip
import json
from pathlib import Path
import tempfile
import unittest

from pepclip.filter_fasta import filter_fasta


class FilterFastaTests(unittest.TestCase):
    def test_compressed_boundaries_duplicates_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.fasta.gz", Path(tmp) / "filtered"
            records = [("short", "A" * 9), ("ten", "A" * 10), ("hundred", "C" * 100),
                       ("long", "A" * 101), ("invalid", "X" * 10), ("duplicate", "A" * 10)]
            with gzip.open(source, "wt") as handle:
                for name, seq in records:
                    handle.write(f">UniRef90_{name} description\n{seq[:5]}\n{seq[5:]}\n")
            metadata = filter_fasta(source, output, "unknown", progress_every=3)
            self.assertEqual(metadata["counts"], dict(total=6, length=2, retained=2,
                                                     nonstandard_residue=1, duplicate_sequence=1))
            self.assertEqual(metadata["retained_length_histogram"], {10: 1, 100: 1})
            self.assertEqual(metadata["splits"], "not assigned")
            self.assertEqual(metadata["graphs"], "not constructed")
            self.assertEqual((output / "peptides.fasta").read_text().count(">"), 2)
            rejections = [json.loads(line) for line in (output / "rejected.jsonl").read_text().splitlines()]
            self.assertEqual(rejections[-1]["duplicate_of"], "UniRef90_ten")
            self.assertEqual(json.loads((output / "metadata.json").read_text())["status"], "complete")
            self.assertFalse((output / "dedup.sqlite3").exists())
            with self.assertRaises(FileExistsError):
                filter_fasta(source, output, "unknown")

    def test_bad_source_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.fasta", Path(tmp) / "filtered"
            source.write_text(">other_database\nAAAAAAAAAA\n")
            with self.assertRaises(ValueError):
                filter_fasta(source, output, "unknown")
            self.assertFalse((output / "metadata.json").exists())
