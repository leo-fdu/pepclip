"""Bridge to the official tencent-ailab/grover checkout; use its own environment."""
from contextlib import nullcontext
from pathlib import Path
import torch
from .base import FrozenEncoder
from ..utils import file_sha256


class GroverEncoder(FrozenEncoder):
    def __init__(self, checkpoint, device="cpu", trusted_checkpoint=False):
        if not trusted_checkpoint:
            raise ValueError("Official GROVER checkpoints contain pickled args; use --trusted-checkpoint only for trusted weights")
        if torch.device(device).type not in {"cpu", "cuda"}:
            raise ValueError("Official GROVER adapter supports CPU/CUDA only")
        from grover.model.models import GroverFpGeneration
        from grover.data.molgraph import MolGraph, BatchMolGraph
        from rdkit import rdBase
        import grover
        # Explicit opt-in: old official checkpoints serialize argparse.Namespace.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = state["args"]
        # Released encoder-only checkpoints can omit finetuning's dropout flag.
        # Dropout is disabled in eval mode regardless; record the default explicitly.
        dropout_defaulted = not hasattr(args, "dropout")
        if dropout_defaulted:
            args.dropout = 0.0
        args.cuda = torch.device(device).type == "cuda"
        args.fingerprint_source = "atom"
        args.embedding_output_type = "atom"
        args.bond_drop_rate = 0
        model = GroverFpGeneration(args)
        required = model.state_dict()
        loaded = state["state_dict"]
        # A deterministic, nonlearned sentinel for empty molecules may be absent
        # from encoder-only checkpoints. Never initialize a missing learned weight.
        constants = {"readout.cached_zero_vector"} - set(loaded)
        parameters = dict(model.named_parameters())
        constants = {k for k in constants if k in required and k in parameters
                     and not parameters[k].requires_grad and not torch.count_nonzero(required[k])}
        missing = [k for k, v in required.items() if k not in constants and (
            k not in loaded or loaded[k].shape != v.shape)]
        if missing:
            raise ValueError(f"GROVER weights missing/incompatible: {missing[:10]}")
        model.load_state_dict({k: required[k] if k in constants else loaded[k]
                               for k in required}, strict=True)
        self.freeze(model, device)
        self.args, self.MolGraph, self.BatchMolGraph = args, MolGraph, BatchMolGraph
        self.output_dim = 2 * args.hidden_size
        if grover.__file__ is not None:
            source_root = Path(grover.__file__).parent
        else:
            # The official checkout uses a namespace package without __init__.py.
            roots = sorted({str(Path(p).resolve()) for p in grover.__path__})
            if len(roots) != 1:
                raise ValueError("Expected one unambiguous GROVER source directory")
            source_root = Path(roots[0])
        self.info = dict(encoder="grover", checkpoint_sha256=file_sha256(checkpoint),
            output_dim=self.output_dim, pooling="mean_atom_from_atom+mean_atom_from_bond",
            extra_features=False, torch_version=str(torch.__version__),
            inference_dropout=args.dropout, dropout_defaulted=dropout_defaulted,
            initialized_nonlearned_constants=sorted(constants),
            rdkit_version=rdBase.rdkitVersion,
            source_sha256={str(p.relative_to(source_root)): file_sha256(p)
                           for p in sorted(source_root.rglob("*.py"))})

    def encode(self, inputs):
        self.model.eval()
        # The upstream implementation uses unqualified .cuda() internally.
        context = torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        with context, torch.no_grad():
            graphs = [self.MolGraph(smiles, self.args) for smiles in inputs]
            batch = self.BatchMolGraph(graphs, self.args).get_components()
            result = self.model(batch, [None] * len(inputs))
        return result.detach().cpu().float()

    def metadata(self):
        return self.info.copy()
