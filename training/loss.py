import torch


def in_pool_contrastive(query: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, tau: float) -> torch.Tensor:
    if positives.ndim == 1:
        positives = positives.unsqueeze(0)
    if negatives.ndim == 1:
        negatives = negatives.unsqueeze(0)
    if negatives.size(0) == 0:
        raise ValueError("need at least one negative")

    pos_logits = (positives @ query) / tau
    neg_logits = (negatives @ query) / tau
    log_num = torch.logsumexp(pos_logits, dim=0)
    log_den = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=0), dim=0)
    return log_den - log_num
