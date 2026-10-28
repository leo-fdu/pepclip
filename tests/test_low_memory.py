"""Bounded input path, exact pairing, projection widths and resume regressions."""
import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pepclip.cache import PairedFeatures
from pepclip.disk_data import iter_ids
from pepclip.model import EmbeddingCLIP, clip_loss
from pepclip.train import (TrainConfig, BufferedBatches, load_inputs, load_checkpoint,
                           model_from_checkpoint, train)
from pepclip.embeddings import import_embeddings
from pepclip.utils import write_json


class LowMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)
        self.ids = [f'id-{i}' for i in range(12)]
        self.paths = []
        for side, dim in [('a', 4), ('b', 7)]:
            raw = self.root / side
            raw.mkdir()
            values = np.arange(12 * dim, dtype=np.float32).reshape(12, dim) / 10
            np.save(raw / 'features.npy', values[::-1].copy())
            write_json(raw / 'ids.json', self.ids[::-1])
            write_json(raw / 'provenance.json', dict(encoder=side, version='test', pooling='test', preprocessing='test'))
            out = self.root / (side + '-bundle')
            import_embeddings(raw / 'features.npy', raw / 'ids.json', raw / 'provenance.json', out)
            self.paths.append(out)
        self.manifest = self.root / 'pairs.jsonl'
        self.records = [dict(sample_id=sid, split='train' if i < 7 else 'val') for i, sid in enumerate(self.ids)]
        self.write_pairs()

    def tearDown(self):
        self.temp.cleanup()

    def write_pairs(self):
        self.manifest.write_text(''.join(json.dumps(row) + '\n' for row in self.records))

    def test_disk_matches_in_memory_pairing_order_and_fingerprint(self):
        normal = load_inputs(self.manifest, *self.paths, TrainConfig())
        disk = load_inputs(self.manifest, *self.paths, TrainConfig(low_memory=1))
        self.assertEqual(normal[3], disk[3])
        self.assertFalse(hasattr(disk[1], 'indices'))
        self.assertFalse(hasattr(disk[1], 'features'))
        for split in ('train', 'val'):
            a, b = PairedFeatures(*normal[:3], split), PairedFeatures(*disk[:3], split)
            self.assertEqual(len(a), len(b))
            for i in range(len(a)):
                for x, y in zip(a[i], b[i]):
                    torch.testing.assert_close(x, y)

    def test_duplicate_missing_and_cross_split_group_rejected(self):
        variants = [self.records + [self.records[0]],
                    self.records + [dict(sample_id='absent', split='train')],
                    [dict(row, group_id='same') for row in self.records],
                    [dict(row, identity_id='same') for row in self.records]]
        for records in variants:
            self.records = records
            self.write_pairs()
            with self.assertRaises(ValueError):
                load_inputs(self.manifest, *self.paths, TrainConfig(low_memory=1))

    def test_stream_json_across_chunks_and_reject_invalid(self):
        path = self.root / 'ids.json'
        ids = ['a' * 65530, 'quote"\\\n', '中文']
        write_json(path, ids)
        self.assertEqual(list(iter_ids(path)), ids)
        for raw in ('[]', '["x",]', '["x"] junk', '[1]', '["x" "y"]'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                list(iter_ids(path))

    def test_bounded_sampler_coverage_determinism_and_singletons(self):
        for size in range(2, 47):
            for batch in (2, 4, 8):
                sampler = BufferedBatches(size, batch, 42, buffer_size=5)
                result = list(sampler)
                self.assertEqual(result, list(sampler))
                self.assertEqual(len(result), len(sampler))
                self.assertEqual(sorted(i for b in result for i in b), list(range(size)))
                self.assertTrue(all(2 <= len(b) <= batch + 1 for b in result))
        self.assertNotEqual(list(BufferedBatches(40, 4, 42, 5)), list(BufferedBatches(40, 4, 43, 5)))

    def test_widths_gradients_checkpoint_and_disk_resume(self):
        model = EmbeddingCLIP(4, 7, hidden_dim_a=13, hidden_dim_b=9, projection_dim=3)
        self.assertEqual(model.sequence_head[0].out_features, 13)
        self.assertEqual(model.graph_head[0].out_features, 9)
        a, b = torch.randn(4, 4), torch.randn(4, 7)
        z, w = model(a, b)
        self.assertEqual(z.shape, (4, 3))
        torch.testing.assert_close(z.norm(dim=1), torch.ones(4))
        clip_loss(z, w).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        config = TrainConfig(hidden_dim_a=13, hidden_dim_b=9, projection_dim=3,
                             batch_size=3, validation_batch_size=2, low_memory=1,
                             shuffle_buffer=5, epochs=2, learnable_temperature=1, save_every=1)
        with contextlib.redirect_stdout(io.StringIO()):
            full = train(self.manifest, *self.paths, self.root / 'full', config)
            partial = train(self.manifest, *self.paths, self.root / 'partial', replace(config, epochs=1))
            resumed = train(self.manifest, *self.paths, self.root / 'resumed', config, partial)
        self.assertTrue((self.root / 'full/epoch_0001.pt').exists())
        snapshot = load_checkpoint(self.root / 'full/epoch_0001.pt')
        self.assertNotIn('optimizer', snapshot)
        self.assertEqual(snapshot['epoch'], 1)
        self.assertTrue(hasattr(model_from_checkpoint(snapshot), 'logit_scale'))
        expected, actual = load_checkpoint(full), load_checkpoint(resumed)
        for key in expected['model']:
            torch.testing.assert_close(expected['model'][key], actual['model'][key], rtol=0, atol=0)
        self.assertEqual(model_from_checkpoint(actual).graph_head[0].out_features, 9)
        # Historical checkpoints omit the new optional widths.
        old = EmbeddingCLIP(4, 7, hidden_dim=8, projection_dim=3)
        restored = model_from_checkpoint(dict(model_args=dict(input_dim_a=4, input_dim_b=7,
                                             hidden_dim=8, projection_dim=3), model=old.state_dict()))
        torch.testing.assert_close(old(a, b)[0], restored(a, b)[0])

    def test_temperature_gradient_bounds_and_evaluation(self):
        from pepclip.evaluate import evaluate
        from torch.utils.data import TensorDataset
        model = EmbeddingCLIP(4, 7, hidden_dim=8, projection_dim=3,
                              learnable_temperature=1, temperature=0.07)
        self.assertAlmostEqual(float(model.loss_temperature().detach()), 0.07, places=6)
        a, b = torch.randn(4, 4), torch.randn(4, 7)
        loss = clip_loss(*model(a, b), model.loss_temperature())
        loss.backward()
        self.assertTrue(torch.isfinite(model.logit_scale.grad))
        self.assertNotEqual(float(model.logit_scale.grad), 0)
        with torch.no_grad():
            model.logit_scale.fill_(100)
        model.clamp_temperature()
        self.assertAlmostEqual(float(model.loss_temperature().detach()), 0.01, places=6)
        expected = clip_loss(*model(a, b), model.loss_temperature()).item()
        actual = evaluate(model, TensorDataset(a, b), batch_size=4, temperature=0.5)
        self.assertAlmostEqual(actual['loss'], expected, places=5)
        with torch.no_grad():
            model.logit_scale.fill_(-100)
        model.clamp_temperature()
        self.assertEqual(float(model.loss_temperature().detach()), 1.0)

    def test_low_memory_retrieval_rejected(self):
        with self.assertRaises(ValueError):
            TrainConfig(low_memory=1, validation_mode='retrieval').validate()


if __name__ == '__main__':
    unittest.main()
