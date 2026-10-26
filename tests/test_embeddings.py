"""Synthetic vector tests; no RDKit, backbone packages, or weights are needed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from pepclip.cache import PairedFeatures
from pepclip.cli import main
from pepclip.embeddings import EmbeddingCache, import_embeddings
from pepclip.data import load_pairs
from pepclip.evaluate import evaluate, validation_batches
from pepclip.inference import ProjectedEmbeddingEncoder
from pepclip.model import EmbeddingCLIP, clip_loss
from pepclip.train import TrainConfig, load_checkpoint, load_inputs, train
from pepclip.utils import write_json


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def bundle(self, name, values, ids):
        raw = self.root / (name + "-raw")
        raw.mkdir()
        np.save(raw / "features.npy", values)
        write_json(raw / "ids.json", ids)
        write_json(raw / "provenance.json", dict(encoder=name, version="synthetic-v1",
            pooling="external test vectors", preprocessing="synthetic; no chemical claims"))
        output = self.root / name
        import_embeddings(raw / "features.npy", raw / "ids.json", raw / "provenance.json", output)
        return output

    def inputs(self):
        ids = [f"sample-{i:02}" for i in range(14)]
        rng = np.random.default_rng(42)
        a, b = rng.normal(size=(14, 4)).astype("float32"), rng.normal(size=(14, 7)).astype("float16")
        cache_a = self.bundle("arbitrary-model-a", a, ids)
        cache_b = self.bundle("arbitrary-model-b", b[::-1], ids[::-1])
        rows = [dict(sample_id=sid, split="train" if i < 7 else "val" if i < 10 else "test",
                     identity_id=f"molecule-{i}") for i, sid in enumerate(ids[:12])]
        pairs = self.root / "pairs.jsonl"
        pairs.write_text("".join(json.dumps(row) + "\n" for row in rows[::-1]))
        return pairs, cache_a, cache_b, a, b

    def test_reordered_superset_bundles_pair_by_id_and_reuse_splits(self):
        pairs, path_a, path_b, a, b = self.inputs()
        rows, ca, cb, provenance = load_inputs(pairs, path_a, path_b, TrainConfig())
        dataset = PairedFeatures(rows, ca, cb, "train")
        torch.testing.assert_close(dataset[0][0], torch.from_numpy(a[0]))
        torch.testing.assert_close(dataset[0][1], torch.from_numpy(b[0].astype("float32")))
        self.assertEqual((ca.output_dim, cb.output_dim), (4, 7))
        self.assertIn("a", provenance)
        # A smaller selection and a changed split do not invalidate immutable bundles.
        subset = [dict(rows[0], split="val"), dict(rows[1], split="val")]
        self.assertEqual(EmbeddingCache(path_a, subset).output_dim, 4)
        self.assertEqual(EmbeddingCache(path_b, subset).output_dim, 7)
        with self.assertRaises(ValueError):
            EmbeddingCache(path_a, [dict(sample_id="missing")])
        with self.assertRaises(ValueError):
            load_inputs(pairs, path_a, path_b, TrainConfig(protein_encoder="esm2"))

    def test_bad_vectors_ids_and_corruption_fail(self):
        for name, values, ids in (
                ("nan", np.array([[np.nan], [1.0]], dtype="float32"), ["x", "y"]),
                ("overflow", np.array([[1e300], [1.0]], dtype="float64"), ["x", "y"]),
                ("integer", np.ones((2, 3), dtype="int32"), ["x", "y"]),
                ("shape", np.ones((2, 3), dtype="float32"), ["x"]),
                ("duplicates", np.ones((2, 3), dtype="float32"), ["x", "x"])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.bundle(name, values, ids)
            self.assertFalse((self.root / name / "metadata.json").exists())
        pairs, path_a, _, _, _ = self.inputs()
        with (path_a / "features.npy").open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaises(ValueError):
            EmbeddingCache(path_a, load_pairs(pairs))

    def test_provenance_required_and_output_never_overwritten(self):
        self.bundle("source", np.ones((2, 3), dtype="float32"), ["x", "y"])
        raw = self.root / "source-raw"
        with self.assertRaises(FileExistsError):
            import_embeddings(raw / "features.npy", raw / "ids.json", raw / "provenance.json", self.root / "source")
        write_json(raw / "provenance.json", {})
        with self.assertRaises(ValueError):
            import_embeddings(raw / "features.npy", raw / "ids.json", raw / "provenance.json", self.root / "bad-source")

    def test_only_projection_parameters_have_gradients(self):
        model = EmbeddingCLIP(4, 7, hidden_dim=12, projection_dim=3)
        a, b = torch.randn(5, 4), torch.randn(5, 7)
        z_a, z_b = model(a, b)
        self.assertEqual(z_a.shape, (5, 3))
        torch.testing.assert_close(z_a.norm(dim=1), torch.ones(5))
        clip_loss(z_a, z_b).backward()
        self.assertEqual(len(list(model.parameters())), 8)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

    def test_batch_validation_is_weighted_fixed_and_not_full_retrieval(self):
        class IdentityPair(torch.nn.Module):
            def forward(self, a, b):
                return a, b
        vectors = torch.eye(5)
        dataset = TensorDataset(vectors, vectors)
        expected = (clip_loss(vectors[:2], vectors[:2]).item() * 2 +
                    clip_loss(vectors[2:], vectors[2:]).item() * 3) / 5
        with patch("pepclip.evaluate.retrieval_metrics", side_effect=AssertionError("No full retrieval")):
            metrics = evaluate(IdentityPair(), dataset, batch_size=2)
        self.assertAlmostEqual(metrics["loss"], expected, places=7)
        self.assertEqual(metrics["sample_count"], 5)
        self.assertEqual((metrics["min_batch_size"], metrics["max_batch_size"]), (2, 3))
        for size in range(2, 20):
            batches = list(validation_batches(size, 4))
            self.assertEqual([i for batch in batches for i in batch], list(range(size)))
            self.assertTrue(all(len(batch) >= 2 for batch in batches))
        retrieval = evaluate(IdentityPair(), dataset, batch_size=2, mode="retrieval")
        self.assertEqual(retrieval["a_to_b"]["recall@1"], 1.0)
        self.assertEqual(retrieval["candidate_count"], 5)

    def test_cli_train_resume_evaluate_and_vector_inference_without_encoders(self):
        pairs, path_a, path_b, a, _ = self.inputs()
        config = TrainConfig(hidden_dim=12, projection_dim=3, batch_size=3,
                             validation_batch_size=2, epochs=2)
        with patch("pepclip.encoders.build_encoder", side_effect=AssertionError("No encoder extraction")), \
                contextlib.redirect_stdout(io.StringIO()):
            full = train(pairs, path_a, path_b, self.root / "full", config)
            config_path = self.root / "config.json"
            write_json(config_path, dict(hidden_dim=12, projection_dim=3, batch_size=3,
                                        validation_batch_size=2, epochs=1))
            main(["train", "--pairs", str(pairs), "--embedding-a", str(path_a),
                  "--embedding-b", str(path_b), "--output", str(self.root / "partial"),
                  "--config", str(config_path)])
            resumed = train(pairs, path_a, path_b, self.root / "resumed", config,
                            self.root / "partial/last.pt")
        expected, actual = load_checkpoint(full), load_checkpoint(resumed)
        for name in expected["model"]:
            torch.testing.assert_close(expected["model"][name], actual["model"][name], rtol=0, atol=0)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            main(["evaluate", "--pairs", str(pairs), "--embedding-a", str(path_a),
                  "--embedding-b", str(path_b), "--checkpoint", str(self.root / "full/best.pt")])
        self.assertEqual(json.loads(captured.getvalue())["evaluation_mode"], "batch")
        encoder = ProjectedEmbeddingEncoder(self.root / "full/best.pt", side="a")
        projected = encoder.encode(a[:2])
        self.assertEqual(projected.shape, (2, 3))
        torch.testing.assert_close(projected.norm(dim=1), torch.ones(2))
        self.assertEqual(encoder.source_metadata["provenance"]["encoder"], "arbitrary-model-a")
        with self.assertRaises(ValueError):
            encoder.encode(np.ones((2, 7), dtype="float32"))
        # Reusing a cache is allowed; resuming with a changed split is not.
        rows = load_pairs(pairs)
        rows[0]["split"] = "val"
        pairs.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaises(ValueError):
            train(pairs, path_a, path_b, self.root / "bad-resume", config, full)


if __name__ == "__main__":
    unittest.main()
