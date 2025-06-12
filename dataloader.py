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

        # 掃描每個資料夾，找出背景、圖層1（葉脈）、圖層3（邊緣）、圖層2（葉面）
        for subdir, _, files in os.walk(root_dir):
            bg_img = vein_img = outline_img = leaf_img = None
            for f in files:
                if "背景" in f:
                    bg_img = os.path.join(subdir, f)
                elif "图层 1" in f:
                    outline_img = os.path.join(subdir, f)
                elif "图层 2" in f:
                    leaf_img = os.path.join(subdir, f)
                elif "图层 3" in f:
                    vein_img = os.path.join(subdir, f)
            if bg_img and vein_img and outline_img and leaf_img:
                self.samples.append((bg_img, vein_img, outline_img, leaf_img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bg_path, vein_path, outline_path, leaf_path = self.samples[idx]

        # 讀取圖片並 resize
        image = Image.open(bg_path).convert("RGB").resize(self.image_size)
        vein = Image.open(vein_path).convert("L").resize(self.image_size)
        outline = Image.open(outline_path).convert("L").resize(self.image_size)
        leaf = Image.open(leaf_path).convert("L").resize(self.image_size)

        # 生成 gt_mask（二值）：黑色線條為 1
        vein_np = np.array(vein)
        outline_np = np.array(outline)
        gt_mask = ((vein_np < 128) | (outline_np < 128)).astype(np.uint8)

        # 生成 leaf_mask（二值）：葉子區域為 1
        leaf_np = np.array(leaf)
        leaf_mask = (leaf_np < 128).astype(np.uint8)

        # 轉為 tensor
        image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255.0
        gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0)
        leaf_mask_tensor = torch.tensor(leaf_mask, dtype=torch.float32).unsqueeze(0)

        return {
            "image": image_tensor,
            "gt_mask": gt_mask_tensor,
            "leaf_mask": leaf_mask_tensor
        }

if __name__ == "__main__":
    # 替換成你自己的資料夾路徑
    dataset_path = "/Users/chiantsii/Desktop/vein_seg/data/LVD2021/36_Holly_labels/train"
    dataset = VeinSegmentationDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 取出一筆資料做測試
    sample = next(iter(dataloader))
    image = sample["image"][0].permute(1, 2, 0).numpy()
    gt_mask = sample["gt_mask"][0][0].numpy()
    leaf_mask = sample["leaf_mask"][0][0].numpy()

    # 顯示圖片
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
