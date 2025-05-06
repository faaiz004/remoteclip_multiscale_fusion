import cv2
import numpy as np
import torch

def rollout_map(model, img_tensor):
    """
    Returns (224,224) numpy heat-map ∈ [0,1] for a single image batch.
    """
    with torch.no_grad():
        x = model.image_enc.stem(img_tensor)
        x = model.image_enc.l1(x); x = model.image_enc.l2(x)
        f3 = model.image_enc.l3(x); f4 = model.image_enc.l4(f3)

        s3 = model.image_enc._seq(model.image_enc.p3(f3))
        s4 = model.image_enc._seq(model.image_enc.p4(f4))
        seq = torch.cat([s3, s4], 1)

        cls = model.image_enc.cls.expand(seq.size(0), -1, -1)
        seq = torch.cat([cls, seq], 1)

        attn = []
        for blk in model.image_enc.blocks:
            seq, w = blk(seq, return_attn=True)
            attn.append(w.mean(1).squeeze(0).cpu().numpy())

        L = attn[0].shape[0]
        R = np.eye(L)
        for A in attn:
            A = A + np.eye(L)
            A /= A.sum(-1, keepdims=True)
            R = A @ R

        mask = R[0, 1:]
        heat = mask[-49:].reshape(7, 7)
        heat = cv2.resize(heat, (224, 224), cv2.INTER_CUBIC)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat
