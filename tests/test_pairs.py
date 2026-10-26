"""Pair-selection checks runnable with the Python standard library only."""
import json
from pathlib import Path
import tempfile
import unittest

from pepclip.data import load_pairs, manifest_fingerprint


class PairManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "pairs.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return load_pairs(self.path)

    def test_generic_ids_no_sequences_and_no_test_split_required(self):
        rows = self.write_rows([dict(sample_id="b", split="val"), dict(sample_id="a", split="train")])
        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["identity_id"], "a")
        self.assertEqual(rows[0]["group_id"], "a")
        self.assertFalse(any("sequence" in row or "smiles" in row for row in rows))

    def test_reordering_stable_but_changed_split_changes_fingerprint(self):
        source = [dict(sample_id="b", split="val"), dict(sample_id="a", split="train")]
        first = self.write_rows(source)
        second = self.write_rows(source[::-1])
        self.assertEqual(first, second)
        self.assertEqual(manifest_fingerprint(first), manifest_fingerprint(second))
        source[0]["split"] = "train"
        self.assertNotEqual(manifest_fingerprint(first), manifest_fingerprint(self.write_rows(source)))

    def test_duplicates_and_invalid_records_rejected(self):
        for rows in (
                [], [dict(sample_id="", split="train")], [dict(sample_id="x", split="invalid")],
                [dict(sample_id="x", split="train"), dict(sample_id="x", split="val")],
                [dict(sample_id="x", split="train", identity_id="same"),
                 dict(sample_id="y", split="val", identity_id="same")],
                [dict(sample_id="x", split="train", group_id="")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.write_rows(rows)

    def test_similarity_groups_cannot_cross_splits(self):
        rows = [dict(sample_id="x", split="train", group_id="family"),
                dict(sample_id="y", split="train", group_id="family")]
        self.assertEqual(len(self.write_rows(rows)), 2)
        rows[1]["split"] = "val"
        with self.assertRaises(ValueError):
            self.write_rows(rows)


if __name__ == "__main__":
    unittest.main()
