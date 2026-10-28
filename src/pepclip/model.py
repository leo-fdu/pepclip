import math
import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, projection_dim):
        if min(input_dim, hidden_dim, projection_dim) < 1:
            raise ValueError("Projection dimensions must be positive")
        super().__init__(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, projection_dim))


class PeptideCLIP(nn.Module):
    def __init__(self, sequence_dim, graph_dim, hidden_dim=512, projection_dim=256,
                 hidden_dim_a=0, hidden_dim_b=0,
                 learnable_temperature=0, temperature=0.07):
        super().__init__()
        if learnable_temperature:
            if not 0.01 <= temperature <= 1.0:
                raise ValueError("Learnable initial temperature must be in [0.01, 1]")
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.sequence_head = ProjectionHead(sequence_dim, hidden_dim_a or hidden_dim, projection_dim)
        self.graph_head = ProjectionHead(graph_dim, hidden_dim_b or hidden_dim, projection_dim)

    def loss_temperature(self, fallback=0.07):
        if hasattr(self, "logit_scale"):
            return self.logit_scale.clamp(0, math.log(100)).neg().exp()
        return fallback

    @torch.no_grad()
    def clamp_temperature(self):
        if hasattr(self, "logit_scale"):
            self.logit_scale.clamp_(0, math.log(100))

    def encode_sequence(self, features):
        return F.normalize(self.sequence_head(features), dim=-1)

    def encode_graph(self, features):
        return F.normalize(self.graph_head(features), dim=-1)

    def forward(self, sequence, graph):
        return self.encode_sequence(sequence), self.encode_graph(graph)


class EmbeddingCLIP(PeptideCLIP):
    """Two independent projection heads over any paired, precomputed vectors.

    Legacy state-dict names are retained for existing peptide checkpoints.
    No pretrained encoder is part of this module or its optimizer.
    """
    def __init__(self, input_dim_a, input_dim_b, hidden_dim=512, projection_dim=256,
                 hidden_dim_a=0, hidden_dim_b=0,
                 learnable_temperature=0, temperature=0.07):
        super().__init__(input_dim_a, input_dim_b, hidden_dim, projection_dim,
                         hidden_dim_a, hidden_dim_b, learnable_temperature, temperature)

    def encode_a(self, features):
        return self.encode_sequence(features)

    def encode_b(self, features):
        return self.encode_graph(features)


def clip_loss(sequence, graph, temperature=0.07):
    if isinstance(temperature, torch.Tensor):
        valid = temperature.numel() == 1 and torch.isfinite(temperature).all() and (temperature > 0).all()
    else:
        valid = math.isfinite(temperature) and temperature > 0
    if not valid:
        raise ValueError("Temperature must be finite and positive")
    if sequence.ndim != 2 or sequence.shape != graph.shape or len(sequence) < 2:
        raise ValueError("CLIP requires matching [B,D] tensors with B >= 2")
    logits = sequence @ graph.T / temperature
    targets = torch.arange(len(sequence), device=sequence.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
