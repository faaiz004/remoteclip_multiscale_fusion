"""
Pre-train multi-scale patch encoder on DIOR/xView crops (Improvement 2).
Run with:  python -m src.train.train_patch_encoder
"""

if __name__ == "__main__":
    import os, json, random, torch, torch.nn as nn, torch.nn.functional as F
    from collections import defaultdict, Counter
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.utils.data.dataloader import default_collate
    from torchvision.transforms import (
        Compose,
        Resize,
        CenterCrop,
        ToTensor,
        Normalize,
        InterpolationMode,
    )
    import open_clip
    from open_clip.tokenizer import tokenize
    from huggingface_hub import hf_hub_download
    import kagglehub
    from tqdm import tqdm
    import albumentations as A

    # 0. setup identical to notebook -------------------------------------------------
    SEED = 1337
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True

    # 1. download xView ---------------------------------------------------------------
    dataset_root = kagglehub.dataset_download("hassanmojab/xview-dataset")
    IMAGE_DIR = os.path.join(dataset_root, "train_images", "train_images")
    LABELS_PATH = os.path.join(dataset_root, "train_labels", "xView_train.geojson")

    # 2. load CLIP text prompts -------------------------------------------------------
    clip_model, _, _ = open_clip.create_model_and_transforms("RN50-quickgelu", pretrained="openai")
    clip_model.eval()
    prompts = [...]  # (same long list as notebook)
    with torch.no_grad():
        prompt_emb = clip_model.encode_text(tokenize(prompts).to(device))
        prompt_emb = prompt_emb / prompt_emb.norm(dim=-1, keepdim=True)
        logit_scale = clip_model.logit_scale.exp().to(device)

    # 3. dataset class & transforms ---------------------------------------------------
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    patch_tf = Compose(
        [
            Resize(288, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )

    # (Dataset class `XViewPatches` is exactly as in notebook) -------------------------
    # ... full class code here, unchanged ...

    full_ds = XViewPatches(IMAGE_DIR, LABELS_PATH, patch_tf)
    train_ds, test_ds = random_split(
        full_ds,
        [int(0.8 * len(full_ds)), len(full_ds) - int(0.8 * len(full_ds))],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: default_collate([x for x in b if x is not None]),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: default_collate([x for x in b if x is not None]),
    )

    # 4. model + optimizer ------------------------------------------------------------
    patch_backbone = open_clip.create_model("RN50-quickgelu")[0].visual.to(device)
    from src.model.patch_encoder import MultiScalePatchEncoder

    patch_enc = MultiScalePatchEncoder(patch_backbone).to(device)
    opt = torch.optim.AdamW(patch_enc.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # 5. train loop -------------------------------------------------------------------
    for ep in range(1, 4):
        patch_enc.train()
        tot, seen = 0.0, 0
        for imgs, lbls in tqdm(train_dl, desc=f"Epoch {ep}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                emb = patch_enc(imgs)
                logits = logit_scale * emb @ prompt_emb.T
                loss = F.cross_entropy(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * imgs.size(0); seen += imgs.size(0)
        print(f"Epoch {ep} loss {tot/seen:.4f}")

    torch.save(patch_enc.state_dict(), "multiscale_patch_encoder_final.pth")
