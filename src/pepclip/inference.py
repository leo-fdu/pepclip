import torch
from .train import load_checkpoint, model_from_checkpoint


class ProjectedEmbeddingEncoder:
    """Apply either trained head to externally computed embeddings; no backbone."""
    def __init__(self, checkpoint, side="b", device="cpu"):
        if side not in {"a", "b"}:
            raise ValueError("side must be a or b")
        state = load_checkpoint(checkpoint)
        self.model = model_from_checkpoint(state, device)
        self.side, self.device = side, device
        key = side if side in state["provenance"] else ("sequence" if side == "a" else "graph")
        self.source_metadata = state["provenance"][key]

    @torch.no_grad()
    def encode(self, features):
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("Provide finite [N,D] precomputed embeddings")
        projection = self.model.sequence_head if self.side == "a" else self.model.graph_head
        if features.shape[1] != projection[0].in_features:
            raise ValueError("Embedding dimension differs from the selected projection head")
        head = self.model.encode_sequence if self.side == "a" else self.model.encode_graph
        return head(features).cpu()


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
