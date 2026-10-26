from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
import torch
from .base import FrozenEncoder
from ..utils import file_sha256


class ESM2Encoder(FrozenEncoder):
    def __init__(self, model_name="esm2_t33_650M_UR50D", checkpoint=None,
                 device="cpu", layer=None):
        import esm
        # Released ESM checkpoints may contain argparse.Namespace. Allow only
        # that specific configuration type under PyTorch 2.6's safe loader.
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        context = safe_globals([Namespace]) if safe_globals else nullcontext()
        with context:
            if checkpoint:
                model, alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint)
                weight_path = Path(checkpoint)
            else:
                if not model_name.startswith("esm2_"):
                    raise ValueError("Expected an ESM2 model name")
                model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
                weight_path = Path(torch.hub.get_dir()) / "checkpoints" / (model_name + ".pt")
        if model.__class__.__name__ != "ESM2":
            raise ValueError("Loaded checkpoint is not an ESM2 encoder")
        self.freeze(model, device)
        self.converter = alphabet.get_batch_converter()
        self.layer = model.num_layers if layer is None else layer
        if not 0 <= self.layer <= model.num_layers:
            raise ValueError("Invalid ESM representation layer")
        self.output_dim = model.embed_dim
        self.info = dict(encoder="esm2", model=model_name, layer=self.layer,
                         pooling="residue_mean_exclude_special_tokens", output_dim=self.output_dim,
                         checkpoint_sha256=file_sha256(weight_path),
                         torch_version=str(torch.__version__),
                         esm_version=getattr(esm, "__version__", "unknown"))

    def encode(self, inputs):
        self.model.eval()
        _, _, tokens = self.converter([(str(i), seq) for i, seq in enumerate(inputs)])
        with torch.no_grad():
            representations = self.model(tokens.to(self.device), repr_layers=[self.layer],
                                         return_contacts=False)["representations"][self.layer]
            pooled = torch.stack([representations[i, 1:len(seq)+1].mean(0)
                                  for i, seq in enumerate(inputs)])
        return pooled.detach().cpu().float()

    def metadata(self):
        return self.info.copy()
