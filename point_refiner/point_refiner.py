import torch
import torch.nn as nn
from .cmm import ConfidenceMaskModule
from .pcm import PointCorrectionModule
from .point_features import PointFeatureExtractor
from .point_head import PointHead
from PIL import Image
import torchvision.transforms as T
import numpy as np


class PointRefiner(nn.Module):
    """
    將 CMM + PCM + PFE + PH 整合為一個模組：
    - Input: decoder 的 coarse mask 與 encoder 的 high-level feature
    - Output: 所有選中的點的 refined logits、座標、點 mask、refined mask、pseudo label Yu
    """
    def __init__(self, top_k=512, top_l=128, top_i=128):
        super().__init__()
        self.cmm = ConfidenceMaskModule(top_k=top_k)
        self.pcm = PointCorrectionModule(top_l=top_l, top_i=top_i)
        self.pfe = PointFeatureExtractor()
        self.ph = PointHead()

    def forward(self, p_coarse_mask, feature_map):
        """
        Args:
            p_coarse_mask: (B, 1, H, W) - decoder output after sigmoid
            feature_map: (B, C, h, w) - from encoder, after DAC+RMP
        Returns:
            logits: list of (K_i, 1) - each point's raw prediction logit
            point_coords: list of (K_i, 2)
            point_mask: Tensor (B, 1, H, W)
            refined_mask: Tensor (B, 1, H, W)
            Yu: Tensor (B, 1, H, W) - final pseudo label
        """
        B, _, H, W = p_coarse_mask.shape

        A, A_unc, Omega_K = self.cmm(p_coarse_mask)
        Omega_L, Omega_I = self.pcm(p_coarse_mask)

        point_coords = []
        for b in range(B):
            all_pts = torch.cat([Omega_K[b], Omega_L[b], Omega_I[b]], dim=0)
            point_coords.append(all_pts)

        point_feats = self.pfe(feature_map, point_coords)
        logits = self.ph(point_feats)

        # === 建立 point mask ===
        point_mask = torch.zeros((B, 1, H, W), device=p_coarse_mask.device)
        for b in range(B):
            if point_coords[b].numel() == 0:
                continue
            coords = point_coords[b].long()
            x = coords[:, 0].clamp(0, W - 1)
            y = coords[:, 1].clamp(0, H - 1)
            point_mask[b, 0, y, x] = 1

        # === 建立 refined mask ===
        refined_mask = p_coarse_mask.clone()
        for b in range(B):
            if point_coords[b].numel() == 0:
                continue
            coords = point_coords[b].long()
            x = coords[:, 0].clamp(0, W - 1)
            y = coords[:, 1].clamp(0, H - 1)
            probs = torch.sigmoid(logits[b].squeeze(1))
            refined_mask[b, 0, y, x] = probs

        # === 建立 A mask，將 refined 點位置設為 0，其餘為 1 ===
        A_mask = torch.ones_like(refined_mask)
        for b in range(B):
            if point_coords[b].numel() == 0:
                continue
            coords = point_coords[b].long()
            x = coords[:, 0].clamp(0, W - 1)
            y = coords[:, 1].clamp(0, H - 1)
            A_mask[b, 0, y, x] = 0

        # === 建立 Yu: 結合 coarse 和 refined 的 pseudo label ===
        Yu = A_mask * p_coarse_mask + refined_mask

        # return logits, point_coords, point_mask, refined_mask, Yu
        return Omega_K, Omega_L, Omega_I

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    from PIL import Image
    from .point_refiner import PointRefiner

    img_path = '/Users/chiantsii/Desktop/vein_seg/predictions/epoch_1/sample_0_coarse.png'

    # === Step 1: 讀入 coarse mask 圖片 ===
    img = Image.open(img_path).convert("L")
    transform = T.ToTensor()
    coarse_mask = transform(img).unsqueeze(0)  # [1,1,H,W]

    refiner = PointRefiner()
    Omega_K, Omega_L, Omega_I = refiner(coarse_mask)



# python -m point_refiner.point_refiner