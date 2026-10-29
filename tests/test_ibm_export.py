import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_ibm_features.py"
spec = importlib.util.spec_from_file_location("ibm_export", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class IBMExportTests(unittest.TestCase):
    def test_rows_keep_duplicates_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            rows = [{"SMILES": "CC", "labels": "1"},
                    {"SMILES": "C", "labels": "0"},
                    {"SMILES": "CC", "labels": "2"}]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["SMILES", "labels"])
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(export.read_rows(path), rows)

    def test_missing_structure_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("sequence,labels\nAAAA,1\n")
            with self.assertRaises(ValueError):
                export.read_rows(path)

    def test_smoke_cannot_write_official_reps(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, str(SCRIPT), "--ibm-root", tmp,
                       "--grover-source", tmp, "--grover-checkpoint", "unused.pt",
                       "--projection-checkpoint", "unused.pt", "--limit", "1",
                       "--output", str(Path(tmp) / "reps")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Smoke exports", result.stderr)
            self.assertFalse((Path(tmp) / "reps").exists())

    def test_runner_rejects_smoke_features_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            reps = Path(tmp) / "reps"
            reps.mkdir()
            (reps / "export.json").write_text(json.dumps({"status": "complete", "smoke": True}))
            result = subprocess.run([sys.executable, str(SCRIPT.with_name("run_ibm_evaluation.py")),
                "--ibm-root", tmp, "--mode", "ood", "--dataset", "cpp",
                "--representation", "pepclip", "--hpo-space", "ibm-yaml"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("non-smoke", result.stderr)


if __name__ == "__main__":
    unittest.main()
