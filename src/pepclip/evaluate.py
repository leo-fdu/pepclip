import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import clip_loss


@torch.no_grad()
def retrieval_metrics(sequence, graph, temperature=0.07, chunk_size=256):
    """Exact full-candidate retrieval, chunked to avoid an N x N allocation.

    Ties receive the worst possible rank (no artificial tie-breaking advantage).
    """
    if sequence.shape != graph.shape or len(sequence) < 2 or chunk_size < 1:
        raise ValueError("Invalid evaluation tensors or chunk size")
    metrics, losses = {}, []
    for name, query, candidates in (("sequence_to_graph", sequence, graph),
                                    ("graph_to_sequence", graph, sequence)):
        ranks, total_loss = [], 0.0
        for start in range(0, len(query), chunk_size):
            logits = query[start:start+chunk_size] @ candidates.T / temperature
            targets = torch.arange(start, start + len(logits), device=logits.device)
            positive = logits[torch.arange(len(logits), device=logits.device), targets]
            rank = (logits >= positive[:, None]).sum(1)
            ranks.append(rank.cpu())
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        ranks = torch.cat(ranks).float()
        metrics[name] = {f"recall@{k}": (ranks <= k).float().mean().item() for k in (1, 5, 10)}
        metrics[name].update(mrr=(1 / ranks).mean().item(), mean_rank=ranks.mean().item())
        losses.append(total_loss / len(query))
    metrics.update(loss=sum(losses) / 2, candidate_count=len(sequence))
    return metrics


def validation_batches(size, batch_size):
    """Fixed-order batches; merge a trailing singleton rather than drop a pair."""
    if size < 2 or batch_size < 2:
        raise ValueError("Validation needs at least two pairs and batch_size >= 2")
    start = 0
    while start < size:
        end = min(start + batch_size, size)
        if size - end == 1:
            end = size
        yield range(start, end)
        start = end


@torch.no_grad()
def evaluate(model, dataset, device="cpu", batch_size=256, temperature=0.07, mode="batch"):
    if mode not in {"batch", "retrieval"}:
        raise ValueError("Evaluation mode must be batch or retrieval")
    model.eval()
    seq, graph, total, count, batches = [], [], 0.0, 0, 0
    sizes = []
    loader = DataLoader(dataset, batch_sampler=validation_batches(len(dataset), batch_size))
    for a, b in loader:
        z_a, z_b = model(a.to(device), b.to(device))
        if mode == "retrieval":
            seq.append(z_a.cpu())
            graph.append(z_b.cpu())
        else:
            total += clip_loss(z_a, z_b, temperature).item() * len(a)
        count += len(a)
        batches += 1
        sizes.append(len(a))
    if mode == "retrieval":
        metrics = retrieval_metrics(torch.cat(seq), torch.cat(graph), temperature, batch_size)
        metrics["a_to_b"] = metrics.pop("sequence_to_graph")
        metrics["b_to_a"] = metrics.pop("graph_to_sequence")
        return dict(metrics, evaluation_mode=mode)
    return dict(loss=total / count, sample_count=count, batch_count=batches,
                batch_size=batch_size, min_batch_size=min(sizes), max_batch_size=max(sizes),
                evaluation_mode=mode)
