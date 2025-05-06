import os
import random
from glob import glob
from typing import List, Dict

import matplotlib.pyplot as plt
from PIL import Image

# ---------------- prompt helper -------------------
def generate_prompt(label: str, template: str = "This is a picture of {}.") -> str:
    return template.format(label)

# ---------------- HF dataset ----------------------
def process_scene_dataset_hf(
    dataset,
    label_names: List[str],
    split: str = "train",
    image_field: str = "image",
    label_field: str = "label",
):
    processed = []
    for example in dataset[split]:
        label_idx = example[label_field]
        label_str = label_names[label_idx]
        prompt = generate_prompt(label_str)
        processed.append(
            {"image": example[image_field], "label": label_str, "prompt": prompt}
        )
    return processed

# --------------- local folder dataset -------------
def load_scene_dataset_local(dataset_path: str):
    data = []
    for class_dir in os.listdir(dataset_path):
        full_dir = os.path.join(dataset_path, class_dir)
        if os.path.isdir(full_dir):
            imgs = glob(os.path.join(full_dir, "*.jpg")) + glob(
                os.path.join(full_dir, "*.png")
            )
            for p in imgs:
                data.append((p, class_dir))
    return data


def process_scene_dataset_local(dataset_path: str):
    data = load_scene_dataset_local(dataset_path)
    out = []
    for img_path, label in data:
        out.append(
            {"image_path": img_path, "label": label, "prompt": generate_prompt(label)}
        )
    return out

# --------------- WHU-RS19 & NWPU helpers ----------
def process_whu_rs19_combined(train_dir: str, val_dir: str):
    data = []
    for folder in [train_dir, val_dir]:
        for cls in os.listdir(folder):
            cpath = os.path.join(folder, cls)
            if not os.path.isdir(cpath):
                continue
            for img in glob(os.path.join(cpath, "*.jpg")):
                data.append(
                    {"image_path": img, "label": cls, "prompt": generate_prompt(cls)}
                )
    return data


def process_nwpu_combined(train_dir: str, test_dir: str):
    data = []
    for folder in [train_dir, test_dir]:
        for cls in os.listdir(folder):
            cpath = os.path.join(folder, cls)
            if not os.path.isdir(cpath):
                continue
            for img in glob(os.path.join(cpath, "*.jpg")):
                data.append(
                    {"image_path": img, "label": cls, "prompt": generate_prompt(cls)}
                )
    return data

# ---------------- visualisation -------------------
def visualize_processed_dataset(data: List[Dict], n: int = 10, title: str = ""):
    samples = random.sample(data, n)
    plt.figure(figsize=(15, 8))
    for idx, s in enumerate(samples):
        img = s.get("image") or Image.open(s["image_path"]).convert("RGB")
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f'{s["label"]}\n{s["prompt"]}', fontsize=8)
    plt.suptitle(title or "Dataset", fontsize=14)
    plt.tight_layout()
    plt.show()
