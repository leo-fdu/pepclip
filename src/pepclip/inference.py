import torch
from .train import load_checkpoint, model_from_checkpoint


class PeptideEncoder:
    """Compose the trained graph projection with an exact frozen graph encoder."""
    def __init__(self, checkpoint, graph_encoder, device="cpu"):
        state = load_checkpoint(checkpoint)
        if state["provenance"]["graph"]["encoder"] != graph_encoder.metadata():
            raise ValueError("Inference graph encoder differs from training cache")
        self.model = model_from_checkpoint(state, device)
        self.encoder, self.device = graph_encoder, device

    @torch.no_grad()
    def encode_molecules(self, smiles):
        # Supports structural SMILES input, without claiming OOD accuracy.
        return self.model.encode_graph(self.encoder.encode(smiles).to(self.device)).cpu()
