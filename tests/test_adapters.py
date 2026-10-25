"""Interface tests with stub upstream modules; no pretrained weights are used."""
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pepclip.encoders.esm2 import ESM2Encoder
from pepclip.encoders.grover import GroverEncoder


class ESM2(nn.Module):
    num_layers = 2
    embed_dim = 1

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, tokens, repr_layers, return_contacts):
        return {"representations": {repr_layers[0]: tokens.float().unsqueeze(-1) * self.scale}}


class AdapterTests(unittest.TestCase):
    def test_esm_pooling_excludes_padding_and_special_tokens(self):
        backbone = ESM2()
        def convert(items):
            max_len = max(len(seq) for _, seq in items)
            tokens = torch.full((len(items), max_len+2), 999)
            for i, (_, seq) in enumerate(items):
                tokens[i, 1:len(seq)+1] = torch.arange(1, len(seq)+1)
            return None, None, tokens
        esm = ModuleType("esm")
        esm.pretrained = SimpleNamespace(load_model_and_alphabet_local=lambda path:
                                         (backbone, SimpleNamespace(get_batch_converter=lambda: convert)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "esm2.pt"
            checkpoint.write_bytes(b"stub-only")
            with patch.dict(sys.modules, {"esm": esm}):
                encoder = ESM2Encoder(checkpoint=checkpoint)
                output = encoder.encode(["AAAAAAAAAA", "CCCCCCCCCCCC"])
                torch.testing.assert_close(output, torch.tensor([[5.5], [6.5]]))
                self.assertFalse(output.requires_grad)
                self.assertFalse(backbone.training)
                self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
                with self.assertRaises(ValueError):
                    ESM2Encoder(checkpoint=checkpoint, layer=3)

    def test_grover_requires_explicit_trust(self):
        with self.assertRaisesRegex(ValueError, "trusted"):
            GroverEncoder("does-not-exist.pt")

    def test_grover_strict_loading_and_fingerprint_contract(self):
        class Fingerprint(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4))
                self.args = args

            def forward(self, batch, extra):
                assert all(value is None for value in extra)
                assert self.args.embedding_output_type == "atom"
                return torch.ones(len(batch), 4) * self.weight

        class MolGraph:
            def __init__(self, smiles, args):
                self.smiles = smiles

        class BatchMolGraph:
            def __init__(self, graphs, args):
                self.graphs = graphs

            def get_components(self):
                return self.graphs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = {name: ModuleType(name) for name in
                       ["grover", "grover.model", "grover.model.models", "grover.data", "grover.data.molgraph"]}
            modules["grover"].__file__ = str(root / "__init__.py")
            (root / "__init__.py").write_text("# test stub\n")
            modules["grover.model.models"].GroverFpGeneration = Fingerprint
            modules["grover.data.molgraph"].MolGraph = MolGraph
            modules["grover.data.molgraph"].BatchMolGraph = BatchMolGraph
            checkpoint = root / "weights.pt"
            state = dict(args=Namespace(hidden_size=2), state_dict={"weight": torch.ones(4)})
            torch.save(state, checkpoint)
            with patch.dict(sys.modules, modules):
                encoder = GroverEncoder(checkpoint, trusted_checkpoint=True)
                self.assertEqual(encoder.encode(["C", "CC"]).shape, (2, 4))
                self.assertFalse(encoder.model.weight.requires_grad)
                state["state_dict"] = {}
                torch.save(state, checkpoint)
                with self.assertRaisesRegex(ValueError, "missing/incompatible"):
                    GroverEncoder(checkpoint, trusted_checkpoint=True)
