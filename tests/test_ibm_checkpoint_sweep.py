import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ibm_checkpoint_sweep as sweep
import ibm_ood_queue as queue

class SweepTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/ibm_checkpoint_sweep.json").read_text())

    def load(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(value))
            return sweep.load_config(path)

    def test_example_and_custom_cartesian_product(self):
        config = self.load(self.config)
        jobs = queue.evaluation_jobs(config["seeds"], config["representations"], config["tasks"])
        self.assertEqual(len(jobs) * len(config["checkpoints"]), 160)
        self.config.update(tasks=["cpp"], seeds=[7, 19], representations=["pepclip"],
                           workers=2, threads_per_worker=4, hpo_trials=12)
        self.config["checkpoints"] = self.config["checkpoints"][:1]
        config = self.load(self.config)
        self.assertEqual(set(queue.evaluation_jobs(config["seeds"], config["representations"], config["tasks"])),
                         {"eval-cpp-07:pepclip", "eval-cpp-19:pepclip"})
        _, _, prepare, evaluate = list(sweep.commands(config))[0]
        self.assertEqual(evaluate[evaluate.index("--tasks")+1], "cpp")
        self.assertEqual(evaluate[evaluate.index("--seed-ids")+1:evaluate.index("--ibm-root")], ["7","19"])
        for flag, value in (("--workers","2"),("--threads","4"),("--trials","12")):
            self.assertEqual(evaluate[evaluate.index(flag)+1], value)
        self.assertEqual(prepare[prepare.index("--threads")+1], "4")

    def test_reject_invalid_configuration(self):
        for key, value in (("tasks", []), ("tasks", ["typo"]), ("seeds", [1,1]),
                           ("seeds", [-1]), ("workers", 0), ("workers", True),
                           ("representations", ["pepclip","pepclip"])):
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    self.load(config)
        for name in ("../escape", ".", "/absolute"):
            config = copy.deepcopy(self.config)
            config["checkpoints"][0]["name"] = name
            with self.assertRaises(ValueError):
                self.load(config)

    def test_dry_run_has_no_side_effects(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(self.config)
            output = Path(tmp) / "must-not-exist"
            config["run"] = str(output)
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, str(ROOT/"scripts/ibm_checkpoint_sweep.py"),
                                     "--config",str(path),"--dry-run"],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["total_evaluations"], 160)
            self.assertFalse(output.exists())

    def test_legacy_job_defaults_unchanged(self):
        self.assertEqual(len(queue.evaluation_jobs(5, ["grover-base","grover-pepclip-concat"])),40)
        self.assertEqual(len(queue.job_specs(25)),109)

if __name__ == "__main__":
    unittest.main()
