"""
Fine-tune Multi-Scale RemoteCLIP on a single P4 (Colab).
Run with:  python -m src.train.train_multiscale
"""

import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip
from huggingface_hub import hf_hub_download

from src.data.data_utils import generate_prompt
from src.model.multiscale_encoder import MultiLevelRemoteCLIP

# --------- replace with your own dataset loading pipeline --------------
# This placeholder assumes lists of dicts with keys:
#   * image (PIL.Image) or image_path (str)
#   * label (str)
#   * prompt (str)
train_data, val_data = [], []  # TODO: load your processed data here
# -----------------------------------------------------------------------

class RemoteSensingDataset(torch.utils.data.Dataset):
    def __init__(self, items, preproc):
        self.items = items
        self.preproc = preproc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = it.get("image") or open_clip.image.open(it["image_path"]).convert("RGB")
        return self.preproc(img), it["label"], it["prompt"]

def collate_fn(batch):
    imgs, labels, prompts = zip(*batch)
    return torch.stack(imgs), list(labels), list(prompts)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_base, _, clip_preproc = open_clip.create_model_and_transforms("RN50")

ckpt = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-RN50.pt")
clip_base.load_state_dict(torch.load(ckpt, map_location="cpu"))

model = MultiLevelRemoteCLIP(clip_base).to(device)
tokenizer = open_clip.get_tokenizer("RN50")

train_loader = DataLoader(
    RemoteSensingDataset(train_data, clip_preproc),
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    RemoteSensingDataset(val_data, clip_preproc),
    batch_size=64,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

opt = torch.optim.AdamW(model.image_enc.parameters(), lr=3e-4)
for epoch in range(1, 4):
    model.train()
    for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(device)
        prompts = [generate_prompt(lbl) for lbl in labels]

        img_f = model.encode_image(imgs)
        with torch.no_grad():
            txt_f = model.encode_text(tokenizer(prompts).to(device))

        logits = model.logit_scale * img_f @ txt_f.T
        loss = F.cross_entropy(logits, torch.arange(len(imgs), device=device))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # ---------- validation ----------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(device)
            prompts = [generate_prompt(lbl) for lbl in labels]
            img_f = model.encode_image(imgs)
            txt_f = model.encode_text(tokenizer(prompts).to(device))
            preds = (img_f @ txt_f.T).argmax(1).cpu()
            correct += (preds == torch.arange(len(imgs))).sum().item()
            total += len(imgs)
    print(f"Epoch {epoch}: Acc = {100*correct/total:.2f}%")

torch.save(model.state_dict(), "multilevel_remoteclip_finetuned.pt")
