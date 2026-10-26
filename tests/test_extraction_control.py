"""CPU-only extraction control regressions, without model/runtime dependencies."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location("extract_embeddings",
    Path(__file__).resolve().parents[1] / "scripts/extract_embeddings.py")
extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extraction)


class ExtractionControlTests(unittest.TestCase):
    def test_spool_scan_tolerates_consumption_during_stat(self):
        deleted, live = Mock(), Mock()
        deleted.stat.side_effect = FileNotFoundError()
        live.stat.return_value.st_size = 123
        directory = Mock()
        directory.glob.return_value = [deleted, live]
        self.assertEqual(extraction.spool_size(directory), 123)

    def test_disjoint_claims_and_committed_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "claims").mkdir()
            (root / "progress/esm2").mkdir(parents=True)
            shards = [{"name":f"{i:06d}","start":i*128,"count":128} for i in range(3)]
            extraction.atomic_json(root / "progress/esm2/000000.json", {"committed":True})
            first = extraction.claim_job(root,"esm2",shards)
            second = extraction.claim_job(root,"esm2",shards)
            self.assertEqual(first[0]["name"],"000001")
            self.assertEqual(second[0]["name"],"000002")
            self.assertIsNone(extraction.claim_job(root,"esm2",shards))


if __name__ == "__main__":
    unittest.main()
