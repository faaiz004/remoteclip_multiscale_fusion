"""
Dual-GPU fusion fine-tune on PatternNet / EuroSAT (Improvement 2).
Run with:  python -m src.train.train_fusion_dual_gpu
Requires 2 visible CUDA devices.
"""

if __name__ == "__main__":
    import os, gc, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from tqdm import tqdm
    import open_clip
    from open_clip.tokenizer import tokenize
    from torchgeo.datasets import PatternNet
    from datasets import load_dataset

    # device placement ----------------------------------------------------
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Requires at least 2 GPUs")
    DEVICE_SCENE = torch.device("cuda:0")
    DEVICE_PATCH = torch.device("cuda:1")
    torch.backends.cuda.matmul.allow_tf32 = True

    # base CLIP -----------------------------------------------------------
    clip_cpu, _, scene_preproc = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained="openai"
    )
    clip_cpu.eval()
    txt_emb_cpu = clip_cpu.encode_text(
        tokenize(PatternNet(root="patternnet", download=True).classes)
    )
    txt_emb_cpu = txt_emb_cpu / txt_emb_cpu.norm(dim=-1, keepdim=True)
    scale_cpu = clip_cpu.logit_scale.exp().clone().detach()

    # move buffers to GPU-0 ----------------------------------------------
    txt_emb = txt_emb_cpu.to(DEVICE_SCENE)
    scale = scale_cpu.to(DEVICE_SCENE)

    # load pre-trained encoders ------------------------------------------
    from src.model.patch_encoder import MultiScalePatchEncoder
    from src.model.multiscale_encoder import MultiScaleEncoder
    from src.model.fusion_head import FusionHead
    from src.model.detector import Detector

    patch_backbone = open_clip.create_model("RN50-quickgelu")[0].visual.to(DEVICE_PATCH)
    scene_backbone = open_clip.create_model("RN50-quickgelu")[0].visual.to(DEVICE_SCENE)

    patch_enc = MultiScalePatchEncoder(patch_backbone).to(DEVICE_PATCH).eval()
    scene_enc = MultiScaleEncoder(scene_backbone).to(DEVICE_SCENE).eval()

    fusion = FusionHead().to(DEVICE_SCENE)  # learnable
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    patch_tf = transforms.Compose(
        [
            transforms.Resize((224, 224), InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )

    model = Detector(
        patch_enc,
        scene_enc,
        fusion,
        txt_emb,
        scale,
        patch_tf,
        patch_size=68,
        chunk_size=16,
    )

    # freeze encoders -----------------------------------------------------
    for p in patch_enc.parameters():
        p.requires_grad = False
    for p in scene_enc.parameters():
        p.requires_grad = False
    for p in fusion.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(fusion.parameters(), lr=1e-4, weight_decay=1e-4)

    # Data: PatternNet ----------------------------------------------------
    full = PatternNet(root="patternnet", download=False)
    tr, va = random_split(
        full,
        [int(0.8 * len(full)), len(full) - int(0.8 * len(full))],
        generator=torch.Generator().manual_seed(42),
    )

    class Wrap(torch.utils.data.Dataset):
        def __init__(self, sub):
            self.sub = sub
            self.tf = scene_preproc

        def __len__(self):
            return len(self.sub)

        def __getitem__(self, i):
            d = self.sub[i]
            img = d["image"].float()
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return self.tf(transforms.ToPILImage()(img)), d["label"].item()

    dl_tr = DataLoader(Wrap(tr), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(Wrap(va), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    best = 0.0
    for ep in range(1, 4):
        model.train(); fusion.train()
        for imgs, lbls in tqdm(dl_tr, desc=f"Train {ep}"):
            imgs, lbls = imgs.to(DEVICE_SCENE), lbls.to(DEVICE_SCENE)
            opt.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            opt.step()

        # validation ------------------------------------------------------
        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, lbls in dl_va:
                imgs, lbls = imgs.to(DEVICE_SCENE), lbls.to(DEVICE_SCENE)
                preds = model(imgs).argmax(1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        acc = correct / total
        print(f"Epoch {ep} val_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(fusion.state_dict(), "patternnet_fused_fusion_only_best.pth")
            print("Saved best fusion head.")

        gc.collect()
        torch.cuda.empty_cache()
