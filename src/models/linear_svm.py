import torch
import torch.nn as nn

class LinearSVM(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            num_classes: int, 
            C: float = 1.0
        ):
        super().__init__()
        self.C = C
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def hinge_loss(self, outputs, targets):
        num_samples = outputs.shape[0]
        correct_scores = outputs[torch.arange(num_samples), targets].view(-1, 1)
        margins = torch.clamp(1 + outputs - correct_scores, min=0)
        margins[torch.arange(num_samples), targets] = 0
        
        loss = torch.mean(torch.sum(margins, dim=1))
        l2_reg = 0.5 * torch.sum(self.linear.weight**2)
        
        return loss + (1/self.C) * l2_reg