import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, d: int = 1024, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d, bias=False),
        )

    def forward(self, a, b):
        if a.device != b.device:
            b = b.to(a.device)
        x = torch.cat([a, b], dim=-1)
        x = self.net(x)
        return x / x.norm(dim=-1, keepdim=True)
