import torch
import torch.nn as nn

# ---------------- residual attention ----------------
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d: int, h: int, r: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d))

    def forward(self, x):
        attn_out, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

# ---------------- multi-scale encoder ---------------
class MultiScaleEncoder(nn.Module):
    def __init__(self, visual_backbone, d: int = 512, layers: int = 2, heads: int = 8):
        super().__init__()
        self.stem = visual_backbone.stem
        self.l1, self.l2 = visual_backbone.layer1, visual_backbone.layer2
        self.l3, self.l4 = visual_backbone.layer3, visual_backbone.layer4

        self.p3 = nn.Conv2d(1024, d, 1)
        self.p4 = nn.Conv2d(2048, d, 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        self.blocks = nn.ModuleList([ResidualAttentionBlock(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1024, bias=False)

    @staticmethod
    def _seq(f):
        b, c, h, w = f.shape
        return f.flatten(2).permute(0, 2, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x)
        f3 = self.l3(x); f4 = self.l4(f3)

        s3 = self._seq(self.p3(f3))
        s4 = self._seq(self.p4(f4))
        seq = torch.cat([s3, s4], dim=1)

        cls_tok = self.cls.expand(x.size(0), -1, -1)
        seq = torch.cat([cls_tok, seq], dim=1)

        for blk in self.blocks:
            seq = blk(seq)

        out = self.ln(seq[:, 0])
        out = self.proj(out)
        return out / out.norm(dim=-1, keepdim=True)

# ---------------- RemoteCLIP wrapper ----------------
class MultiLevelRemoteCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip = clip_model
        self.image_enc = MultiScaleEncoder(self.clip.visual)

    def encode_image(self, x):
        return self.image_enc(x)

    def encode_text(self, tokens):
        with torch.no_grad():
            return self.clip.encode_text(tokens)

    @property
    def logit_scale(self):
        return self.clip.logit_scale.exp()
