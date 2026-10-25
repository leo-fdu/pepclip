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
    def __init__(self, sequence_dim, graph_dim, hidden_dim=512, projection_dim=256):
        super().__init__()
        self.sequence_head = ProjectionHead(sequence_dim, hidden_dim, projection_dim)
        self.graph_head = ProjectionHead(graph_dim, hidden_dim, projection_dim)

    def encode_sequence(self, features):
        return F.normalize(self.sequence_head(features), dim=-1)

    def encode_graph(self, features):
        return F.normalize(self.graph_head(features), dim=-1)

    def forward(self, sequence, graph):
        return self.encode_sequence(sequence), self.encode_graph(graph)


def clip_loss(sequence, graph, temperature=0.07):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if sequence.ndim != 2 or sequence.shape != graph.shape or len(sequence) < 2:
        raise ValueError("CLIP requires matching [B,D] tensors with B >= 2")
    logits = sequence @ graph.T / temperature
    targets = torch.arange(len(sequence), device=sequence.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
