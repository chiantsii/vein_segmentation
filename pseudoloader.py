import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class PseudoLabelDataset(Dataset):
    def __init__(self, pseudo_root, image_size=(448, 448)):
        self.image_size = image_size
        self.samples = []

        for sub in sorted(os.listdir(pseudo_root)):
            folder = os.path.join(pseudo_root, sub)
            if os.path.isdir(folder):
                img_path = os.path.join(folder, "image.png")
                mask_path = os.path.join(folder, "pseudo.png")
                bmap_path = os.path.join(folder, "b_map.png")
                if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(bmap_path):
                    self.samples.append((img_path, mask_path, bmap_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, bmap_path = self.samples[idx]

        # 1. input image
        image = Image.open(img_path).convert("RGB").resize(self.image_size)
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0  # [3, H, W]

        # 2. mask & b_map (single-channel)
        mask = Image.open(mask_path).convert("L").resize(self.image_size)
        b_map = Image.open(bmap_path).convert("L").resize(self.image_size)

        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0  # [1, H, W]
        b_map = torch.from_numpy(np.array(b_map)).float().unsqueeze(0) / 255.0

        return {
            "image": image,       # torch.Size([3, H, W])
            "pseudo_mask": mask,      # torch.Size([1, H, W])
            "b_map": b_map        # torch.Size([1, H, W])
        }

