import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch import nn

from pepclip.cache import extract, FeatureCache, PairedFeatures
from pepclip.cli import main
from pepclip.data import prepare, load_manifest, sequence_to_smiles, GRAPH_POLICY
from pepclip.encoders.base import FrozenEncoder
from pepclip.evaluate import retrieval_metrics
from pepclip.inference import PeptideEncoder
from pepclip.model import PeptideCLIP, clip_loss
from pepclip.train import TrainConfig, ContrastiveBatches, train, load_checkpoint, model_from_checkpoint
from pepclip.utils import file_sha256, write_json

# Unit tests deliberately use synthetic encoders; never claim pretrained-model validation.
class ToyEncoder(FrozenEncoder):
    output_dim = 6

    def __init__(self, name):
        self.name = name
        torch.manual_seed(7)
        self.freeze(nn.Sequential(nn.Linear(10, 6), nn.Dropout(0.9)), "cpu")

    def encode(self, inputs):
        self.model.eval()
        x = torch.tensor([[ord(c) / 100 for c in text[:10]] for text in inputs])
        with torch.no_grad():
            return self.model(x).detach().cpu()

    def metadata(self):
        return dict(encoder=self.name, output_dim=self.output_dim, synthetic_test_only=True)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def inputs(self):
        residues = "ACDEFGHIKLMNPQRSTVWY"
        rows = []
        for i in range(12):
            seq = residues[i] + "ACDEFGHIK"
            rows.append(dict(sample_id=f"UniRef90_{i}", sequence=seq,
                smiles=sequence_to_smiles(seq), graph_policy=GRAPH_POLICY,
                group_id=str(i), split="train" if i < 8 else "val" if i < 10 else "test"))
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        seq, graph = self.root / "seq", self.root / "graph"
        with contextlib.redirect_stdout(io.StringIO()):
            extract(manifest, seq, ToyEncoder("esm2"), "sequence", 3)
            extract(manifest, graph, ToyEncoder("grover"), "graph", 4)
        return manifest, seq, graph, rows

    def test_prepare_boundaries_gzip_and_rejections(self):
        fasta = self.root / "input.fasta.gz"
        with gzip.open(fasta, "wt") as f:
            f.write(">UniRef90_a\nACDEF\nGHIKL\n>UniRef90_b\n" + "A"*100 +
                    "\n>UniRef90_c\n" + "A"*101 + "\n>UniRef90_d\nACDEFGHIKX\n" +
                    ">UniRef90_e\nACDEFGHIKL\n>UniRef90_f\nAAAAAAAAA\n")
        counts = prepare(fasta, self.root / "prepared", "test-release")
        self.assertEqual(counts["length"], 2)
        self.assertEqual(counts["nonstandard_residue"], 1)
        self.assertEqual(counts["duplicate_sequence"], 1)
        rows = load_manifest(self.root / "prepared/manifest.jsonl")
        self.assertEqual([len(r["sequence"]) for r in rows], [10, 100])
        with self.assertRaises(FileExistsError):
            prepare(fasta, self.root / "prepared", "test-release")

    def test_graph_chemistry(self):
        mol = Chem.MolFromSmiles(sequence_to_smiles("ACDEFGHIKL"))
        self.assertEqual(Chem.GetFormalCharge(mol), 0)
        # Nine alpha centers (glycine is achiral), plus isoleucine's side chain.
        self.assertEqual(len(Chem.FindMolChiralCenters(mol)), 10)
        self.assertEqual(len(Chem.GetMolFrags(mol)), 1)
        # Sum of free amino-acid formulas minus nine waters for peptide bonds.
        self.assertEqual(rdMolDescriptors.CalcMolFormula(mol), "C50H77N13O15S")
        with self.assertRaises(ValueError):
            sequence_to_smiles("ACDX")
        cysteines = Chem.MolFromSmiles(sequence_to_smiles("CCCCCCCCCC"))
        self.assertFalse(cysteines.HasSubstructMatch(Chem.MolFromSmarts("[#16]-[#16]")))

    def test_external_groups_keep_related_sequences_together(self):
        fasta = self.root / "in.fa"
        fasta.write_text(">UniRef90_a\nACDEFGHIKL\n>UniRef90_b\nACDEFGHIKM\n")
        groups = self.root / "groups.csv"
        groups.write_text("sample_id,group_id\nUniRef90_a,family\nUniRef90_b,family\n")
        prepare(fasta, self.root / "out", "test", groups=groups)
        rows = load_manifest(self.root / "out/manifest.jsonl")
        self.assertEqual(rows[0]["split"], rows[1]["split"])

    def test_symmetric_loss_and_retrieval(self):
        x = torch.eye(8)
        good = clip_loss(x, x)
        bad = clip_loss(x, x.roll(1, 0))
        self.assertLess(good, bad)
        self.assertAlmostEqual(clip_loss(x, x.roll(1, 0)).item(),
                               clip_loss(x.roll(1, 0), x).item())
        metrics = retrieval_metrics(x, x, chunk_size=3)
        self.assertEqual(metrics["sequence_to_graph"]["recall@1"], 1)
        self.assertAlmostEqual(metrics["loss"], good.item(), places=6)
        self.assertEqual(retrieval_metrics(torch.ones(8, 3), torch.ones(8, 3))[
            "sequence_to_graph"]["recall@1"], 0)
        with self.assertRaises(ValueError):
            clip_loss(x[:1], x[:1])

    def test_projection_trains_and_frozen_backbone_stays_fixed(self):
        encoder = ToyEncoder("test")
        before = {k: v.clone() for k, v in encoder.model.state_dict().items()}
        features = encoder.encode(["ACDEFGHIKL", "CDEFGHIKLM", "DEFGHIKLMN"])
        self.assertTrue(torch.equal(features, encoder.encode(["ACDEFGHIKL", "CDEFGHIKLM", "DEFGHIKLMN"])))
        model = PeptideCLIP(6, 6, 16, 4)
        z, g = model(features, features)
        clip_loss(z, g).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in encoder.model.state_dict().items()))
        torch.testing.assert_close(z.norm(dim=1), torch.ones(3))

    def test_cache_pairs_by_id_not_row(self):
        manifest, seq_path, graph_path, rows = self.inputs()
        features = np.load(graph_path / "features.npy").copy()[::-1]
        np.save(graph_path / "features.npy", features)
        ids = json.loads((graph_path / "ids.json").read_text())[::-1]
        write_json(graph_path / "ids.json", ids)
        meta = json.loads((graph_path / "metadata.json").read_text())
        meta.update(features_sha256=file_sha256(graph_path / "features.npy"),
                    ids_sha256=file_sha256(graph_path / "ids.json"))
        write_json(graph_path / "metadata.json", meta)
        seq = FeatureCache(seq_path, rows, "sequence", "esm2")
        graph = FeatureCache(graph_path, rows, "graph", "grover")
        dataset = PairedFeatures(rows, seq, graph, "train")
        torch.testing.assert_close(dataset[0][1], torch.from_numpy(features[-1].copy()))
        with self.assertRaises(ValueError):
            FeatureCache(seq_path, rows, "sequence", "other")
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            FeatureCache(seq_path, rows, "sequence", "esm2")

    def test_corrupted_cache_fails(self):
        _, seq, _, rows = self.inputs()
        with (seq / "features.npy").open("ab") as f:
            f.write(b"corruption")
        with self.assertRaises(ValueError):
            FeatureCache(seq, rows, "sequence", "esm2")

    def test_batches_do_not_drop_samples_or_create_singletons(self):
        for size in range(2, 15):
            batches = list(ContrastiveBatches(size, 4, 42))
            self.assertTrue(all(len(b) >= 2 for b in batches))
            self.assertEqual(sorted(sum(batches, [])), list(range(size)))

    def test_small_dataset_overfit(self):
        torch.manual_seed(8)
        a, b = torch.randn(12, 8), torch.randn(12, 10)
        model = PeptideCLIP(8, 10, 32, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        first = clip_loss(*model(a, b)).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = clip_loss(*model(a, b))
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first * 0.05)
        self.assertEqual(retrieval_metrics(*model(a, b))["graph_to_sequence"]["recall@1"], 1)

    def test_cli_config_override_and_graph_inference(self):
        manifest, seq, graph, _ = self.inputs()
        config = self.root / "config.json"
        write_json(config, dict(hidden_dim=8, projection_dim=3, batch_size=4, epochs=1))
        out = self.root / "cli"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["train", "--manifest", str(manifest), "--sequence-cache", str(seq),
                  "--graph-cache", str(graph), "--output", str(out), "--config", str(config),
                  "--hidden-dim", "17", "--projection-dim", "5"])
        state = load_checkpoint(out / "best.pt")
        self.assertEqual(state["model_args"]["hidden_dim"], 17)
        encoder = PeptideEncoder(out / "best.pt", ToyEncoder("grover"))
        output = encoder.encode_molecules([sequence_to_smiles("ACDEFGHIKL")])
        self.assertEqual(output.shape, (1, 5))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(1))
        with self.assertRaises(ValueError):
            PeptideEncoder(out / "best.pt", ToyEncoder("wrong"))

    def test_resume_matches_uninterrupted_and_cli_evaluation(self):
        manifest, seq, graph, _ = self.inputs()
        config = TrainConfig(hidden_dim=12, projection_dim=4, batch_size=3, epochs=2)
        with contextlib.redirect_stdout(io.StringIO()):
            full = train(manifest, seq, graph, self.root / "full", config)
            config.epochs = 1
            partial = train(manifest, seq, graph, self.root / "partial", config)
            config.epochs = 2
            resumed = train(manifest, seq, graph, self.root / "resumed", config, partial)
        a, b = load_checkpoint(full), load_checkpoint(resumed)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        model = model_from_checkpoint(a)
        torch.testing.assert_close(model.encode_graph(torch.ones(2, 6)),
                                   model_from_checkpoint(b).encode_graph(torch.ones(2, 6)))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            main(["evaluate", "--manifest", str(manifest), "--sequence-cache", str(seq),
                  "--graph-cache", str(graph), "--checkpoint", str(self.root / "full/best.pt")])
        self.assertEqual(json.loads(out.getvalue())["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
