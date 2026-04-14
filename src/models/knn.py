import torch
import torch.nn as nn

class KNNClassifier(nn.Module):
    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k
        self.register_buffer('x_train', torch.empty(0))
        self.register_buffer('y_train', torch.empty(0, dtype=torch.long))

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor):
        self.register_buffer('x_train', x_train)
        self.register_buffer('y_train', y_train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.x_train, p=2)
        _, idx = torch.topk(dists, self.k, largest=False)
        labels = self.y_train[idx]
        preds, _ = torch.mode(labels, dim=1)
        return preds