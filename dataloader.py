import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

class VeinSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(448, 448)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.samples = []

        # **在這裡新增兩個屬性，預設為空**
        self.masks = None
        self.b_maps = None

        # 掃描檔案夾，填 self.samples …
        for subdir, _, files in os.walk(root_dir):
            bg_img = vein_img = outline_img = leaf_img = None
            for f in files:
                if "背景" in f:        bg_img = os.path.join(subdir, f)
                elif "图层 1" in f:   outline_img = os.path.join(subdir, f)
                elif "图层 2" in f:   leaf_img    = os.path.join(subdir, f)
                elif "图层 3" in f:   vein_img    = os.path.join(subdir, f)
            if bg_img and vein_img and outline_img and leaf_img:
                self.samples.append((bg_img, vein_img, outline_img, leaf_img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bg_path, vein_path, outline_path, leaf_path = self.samples[idx]

        # 讀圖與產生 image_tensor, gt_mask_tensor, leaf_mask_tensor 如你原本
        image = Image.open(bg_path).convert("RGB").resize(self.image_size)
        vein = Image.open(vein_path).convert("L").resize(self.image_size)
        outline = Image.open(outline_path).convert("L").resize(self.image_size)
        leaf = Image.open(leaf_path).convert("L").resize(self.image_size)

        vein_np = np.array(vein)
        outline_np = np.array(outline)
        gt_mask = ((vein_np < 128) | (outline_np < 128)).astype(np.uint8)
        leaf_np = np.array(leaf)
        leaf_mask = (leaf_np < 128).astype(np.uint8)

        image_tensor      = torch.tensor(np.array(image).transpose(2,0,1), dtype=torch.float32)/255.0
        gt_mask_tensor    = torch.tensor(gt_mask,   dtype=torch.float32).unsqueeze(0)
        leaf_mask_tensor  = torch.tensor(leaf_mask, dtype=torch.float32).unsqueeze(0)

        sample = {
            "image": image_tensor,
            "gt_mask": gt_mask_tensor,
            "leaf_mask": leaf_mask_tensor
        }

        # **如果已經建立過 pseudo-label，就回傳 mask & b_map**
        if self.masks is not None and self.b_maps is not None:
            # self.masks[idx] 應是張量 (H,W) or (1,H,W)
            m = self.masks[idx]
            b = self.b_maps[idx]
            # 如果沒加 channel dim，就補上
            if m.ndim == 2: m = m[None]
            if b.ndim == 2: b = b[None]
            sample["mask"]  = torch.tensor(m, dtype=torch.float32)
            sample["b_map"] = torch.tensor(b, dtype=torch.float32)
        # 否則，用真實標註當 supervision，並且 confidence 全 1
        else:
            sample["mask"]  = gt_mask_tensor
            sample["b_map"] = torch.ones_like(gt_mask_tensor)

        return sample


if __name__ == "__main__":

    which       = '36_Holly_labels'
    DATA_ROOT = r"C:\Users\qmwn1\OneDrive\桌面\vein_segmentation-main\data\LVD2021" + "\\" + which + r"\pretrain"
    dataset = VeinSegmentationDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sample = next(iter(dataloader))
    image = sample["image"][0].permute(1, 2, 0).numpy()
    gt_mask = sample["gt_mask"][0][0].numpy()
    leaf_mask = sample["leaf_mask"][0][0].numpy()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Vein + Outline Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(leaf_mask, cmap="gray")
    plt.title("Leaf Area Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# python dataloader.py 測試 ｏｋ