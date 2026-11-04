import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ibm_ood_queue.py"
spec = importlib.util.spec_from_file_location("ibm_queue", SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class IBMQueueTests(unittest.TestCase):
    def test_combined_jobs(self):
        jobs = queue.evaluation_jobs(5, ["grover-base", "grover-pepclip-concat"])
        self.assertEqual(len(jobs), 40)
        self.assertIn("eval-binding-04:grover-pepclip-concat", jobs)
        self.assertEqual(sum(j.endswith(":grover-base") for j in jobs), 20)

    def test_concat_preserves_values_and_order(self):
        import pickle
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw = np.array([[1, 2], [3, 4]], dtype=np.float32)
            aligned = np.array([[0.5], [-0.5]], dtype=np.float32)
            for name, array in [("grover-base", raw), ("pepclip", aligned)]:
                with (folder / f"{name}_fixture.pickle").open("wb") as f:
                    pickle.dump(array, f)
            actual = queue.load_features(folder, "fixture", "grover-pepclip-concat")
            np.testing.assert_array_equal(actual, [[1, 2, 0.5], [3, 4, -0.5]])
            np.testing.assert_array_equal(queue.load_features(folder, "fixture", "grover-base"), raw)

    def test_seed_jobs_and_dependencies(self):
        jobs = queue.job_specs(25)
        self.assertEqual(len(jobs), 109)
        self.assertEqual(sum(j.startswith("eval-") for j in jobs), 100)
        self.assertEqual(jobs["eval-binding-24"], ["feature-c-binding", "feature-nc-binding", "targets"])
        self.assertEqual(jobs["eval-cpp-00"], ["feature-c-cpp", "feature-nc-cpp"])

    def test_resume_checks_completed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = root / "jobs/example/attempt-001"
            attempt.mkdir(parents=True)
            output = attempt / "result.json"
            output.write_text('{"value": 1}')
            queue.mark_done(root, "example", attempt, [output])
            self.assertIsNotNone(queue.completed(root, "example"))
            output.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                queue.completed(root, "example")

    def test_missing_marker_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(queue.completed(Path(tmp), "example"))

    def test_cached_inputs_reject_changed_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "downstream_data").mkdir()
            for job in queue.job_specs(0):
                attempt = root / "jobs" / job / "attempt-001"
                attempt.mkdir(parents=True)
                if job == "targets":
                    output = attempt / "targets.json"
                    output.write_text('{}')
                    queue.mark_done(root, job, attempt, [output])
                    continue
                dataset = job.removeprefix("feature-")
                source = root / "downstream_data" / f"{dataset}.csv"
                source.write_text('SMILES,labels\nC,1\n')
                folder = attempt / "reps"
                folder.mkdir()
                feature = folder / f"grover-base_{dataset}.pickle"
                feature.write_bytes(b'fixture')
                report = folder / "export.json"
                report.write_text(json.dumps(dict(status="complete", smoke=False,
                    datasets={dataset: dict(source_rows=1, exported_rows=1,
                        source_sha256=queue.sha(source),
                        files={"grover-base": dict(sha256=queue.sha(feature))})})))
                queue.mark_done(root, job, attempt, [feature, report])
            self.assertTrue(queue.cached_inputs(root, root, "grover-base"))
            source.write_text('SMILES,labels\nC,0\n')
            with self.assertRaisesRegex(ValueError, "changed cached dataset"):
                queue.cached_inputs(root, root, "grover-base")


if __name__ == "__main__":
    unittest.main()
