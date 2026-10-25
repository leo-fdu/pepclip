from abc import ABC, abstractmethod
import torch


class FrozenEncoder(ABC):
    """Extraction-only adapter; not a child of the trainable CLIP module."""
    def freeze(self, model, device):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)

    @abstractmethod
    def encode(self, inputs):
        """Return a detached ordinary CPU float32 tensor [B,D]."""

    @abstractmethod
    def metadata(self):
        """Identify the exact encoder, weights, and preprocessing."""
