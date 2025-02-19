import torch
from torch import nn


class NegPearson(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, predictions, ground_truths):
        predictions = predictions if predictions.ndim == 2 else predictions.squeeze(-1)
        ground_truths = ground_truths if ground_truths.ndim == 2 else ground_truths.squeeze(-1)
        pearson = self.cos(predictions - predictions.mean(dim=1, keepdim=True), ground_truths - ground_truths.mean(dim=1, keepdim=True))
        return torch.mean(-pearson + 1)
    
