import torch
import torch.nn as nn

class SAB(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        a = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + a
        return x + self.mlp(self.ln2(x))


class MultiScalePatchEncoder(nn.Module):
    def __init__(
        self,
        visual_backbone,
        dim: int = 512,
        heads: int = 8,
        l2_blocks: int = 1,
        l34_blocks: int = 2,
    ):
        super().__init__()
        self.stem = visual_backbone.stem
        self.l1, self.l2 = visual_backbone.layer1, visual_backbone.layer2
        self.l3, self.l4 = visual_backbone.layer3, visual_backbone.layer4

        self.p2 = nn.Conv2d(512, dim, 1)
        self.p3 = nn.Conv2d(1024, dim, 1)
        self.p4 = nn.Conv2d(2048, dim, 1)

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.sa2 = nn.ModuleList([SAB(dim, heads) for _ in range(l2_blocks)])
        self.sa34 = nn.ModuleList([SAB(dim, heads) for _ in range(l34_blocks)])

        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, visual_backbone.output_dim, bias=False)

    @staticmethod
    def flat(f):
        b, c, h, w = f.shape
        return f.view(b, c, h * w).permute(0, 2, 1)

    def forward(self, x):
        h2 = self.l2(self.l1(self.stem(x)))
        h3, h4 = self.l3(h2), self.l4(h3 := self.l3(h2))

        t2 = self.flat(self.p2(h2))
        for blk in self.sa2:
            t2 = blk(t2)

        t3 = self.flat(self.p3(h3))
        t4 = self.flat(self.p4(h4))

        seq = torch.cat([self.cls.expand(t2.size(0), -1, -1), t2, t3, t4], dim=1)
        for blk in self.sa34:
            seq = blk(seq)

        out = self.proj(self.ln(seq[:, 0]))
        return out / out.norm(dim=-1, keepdim=True)
