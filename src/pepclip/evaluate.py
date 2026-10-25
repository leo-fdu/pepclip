import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


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


@torch.no_grad()
def evaluate(model, dataset, device="cpu", batch_size=256, temperature=0.07):
    model.eval()
    seq, graph = [], []
    for a, b in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        z_a, z_b = model(a.to(device), b.to(device))
        seq.append(z_a.cpu())
        graph.append(z_b.cpu())
    return retrieval_metrics(torch.cat(seq), torch.cat(graph), temperature, batch_size)
