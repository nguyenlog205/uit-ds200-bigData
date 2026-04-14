# capsule_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, 8)
        return squash(out)

class DigitCaps(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, routing_iters=3):
        super().__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(1, in_caps, out_caps, out_dim, in_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        u_hat = torch.matmul(self.W, x).squeeze(4)
        b = torch.zeros(batch_size, self.in_caps, self.out_caps, 1, device=x.device)
        for _ in range(self.routing_iters):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = squash(s)
            if _ < self.routing_iters - 1:
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)
        return v.squeeze(1)

def squash(x, dim=-1):
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    scale = norm_sq / (1 + norm_sq)
    return scale * x / torch.sqrt(norm_sq + 1e-8)

class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes=50, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCaps(256, 256, kernel_size=9, stride=2)
        self.digit_caps = DigitCaps(in_caps=32*6*6, in_dim=8, out_caps=num_classes, out_dim=16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return torch.norm(x, dim=-1)