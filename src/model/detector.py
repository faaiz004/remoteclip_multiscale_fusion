import torch
import torch.nn as nn

class Detector(nn.Module):
    """
    Dual-branch model:
      * scene encoder on GPU-0
      * patch encoder on GPU-1
      * fusion head on GPU-0
    """

    def __init__(
        self,
        patch_enc: nn.Module,
        scene_enc: nn.Module,
        fusion: nn.Module,
        text_emb: torch.Tensor,
        scale_val: torch.Tensor,
        patch_tf,
        patch_size: int = 68,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.penc = patch_enc
        self.senc = scene_enc
        self.fuse = fusion

        self.register_buffer("text_emb", text_emb, persistent=False)
        self.register_buffer("scale", scale_val, persistent=False)

        self.patch_tf = patch_tf
        self.k = patch_size
        self.chunk = chunk_size

    # ------------ helper: central 4 patches -----------------------------
    def _get_patches(self, x):
        k = self.k
        b, _, h, w = x.shape
        nh, nw = h // k, w // k
        if nh < 3 or nw < 3:
            return None
        P = x.unfold(2, k, k).unfold(3, k, k)
        mid_h, mid_w = nh // 2, nw // 2
        idx = [(mid_h - 1, mid_w), (mid_h, mid_w - 1), (mid_h, mid_w),
               (mid_h, mid_w + 1)]
        valid = [P[:, r, c] for r, c in idx if 0 <= r < nh and 0 <= c < nw]
        return torch.stack(valid, dim=1) if valid else None

    # ------------ forward -----------------------------------------------
    def forward(self, x):
        g = self.senc(x)  # scene embedding on GPU-0

        patches = self._get_patches(x)
        if patches is None:
            p_emb = torch.zeros_like(g)
        else:
            b, n, c, ph, pw = patches.shape
            flat = patches.view(b * n, c, ph, pw)
            flat_tf = self.patch_tf(flat).to(next(self.penc.parameters()).device)

            # encode in manageable chunks to save VRAM
            outs = []
            for i in range(0, flat_tf.size(0), self.chunk):
                outs.append(self.penc(flat_tf[i : i + self.chunk]))
            p_emb = torch.cat(outs, 0).view(b, n, -1).mean(1).to(g.device)

        fused = self.fuse(g, p_emb)
        return self.scale * fused @ self.text_emb.T
