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


if __name__ == "__main__":
    unittest.main()
