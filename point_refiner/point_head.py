import torch
import torch.nn as nn
import torch.nn.functional as F

class PointHead(nn.Module):
    """
    PH: Point Head
    - 將每個點的特徵向量輸入到 MLP 中，預測其是否為前景（vein）點
    - 最終輸出是 sigmoid 機率值（BCE Loss 可用）
    """
    def __init__(self, input_dim=516):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output: logit
        )

    def forward(self, point_features):
        """
        Args:
            point_features: List[Tensor (K_i, C)] - 每張圖的點特徵
        Returns:
            logits: List[Tensor (K_i, 1)] - 每張圖的每個點的預測
        """
        return self.mlp(point_features)


import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from encoder import Encoder
from decoder import Decoder
from core_net import CoRE_Net

if __name__ == '__main__':

    from point_refiner.point_refiner import PointRefiner
    # ======= 初始化模型與模組 =======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 1. 參數設定 ====
    image_path = r"data/LVD2021/36_Holly_labels/train/36_2/2_0003_背景.jpg"
    ckpt_path  = "checkpoints/36_Holly_labels/best_model.pth"

    # ==== 2. 載入圖片 & 預處理 ====
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # ==== 3. 載入模型 ====
    model = CoRE_Net(Encoder(), Decoder(), PointRefiner()).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ======= forward 到 coarse mask =======
    with torch.no_grad():
        out = model(img_tensor, use_refiner=False)
        coarse_mask = out["coarse_logits"]  # shape: (1, 1, H, W)
        feat = out["features"]              # encoder features

        # 抽出 confidence mask 和抽樣點 Ω_K
        A, _, Omega_K = model.point_refiner.cmm(coarse_mask)
        Omega_L, Omega_I = model.point_refiner.pcm(coarse_mask)
        print(f"✅ CMM 抽出點數 Ω_K: {len(Omega_K)}")
        print(f"✅ PCM 抽出點數 Ω_L: {len(Omega_L)} / Ω_I: {len(Omega_I)}")

        # Ω_K 可能是 List[Tensor]，需轉為單一 tensor
        if isinstance(Omega_K, list):
            Omega_K = torch.cat(Omega_K, dim=0)
        # 確保 Ω_L, Ω_I 都是 Tensor
        if isinstance(Omega_L, list):
            Omega_L = torch.cat(Omega_L, dim=0)
        if isinstance(Omega_I, list):
            Omega_I = torch.cat(Omega_I, dim=0)
            
        def print_point_shapes(name, points):
            print(f"\n{name} 抽出點總數: {len(points) if isinstance(points, list) else 'Tensor'}")
            if isinstance(points, list):
                for i, pt in enumerate(points[:10]):
                    print(f"  {name}[{i}] shape: {pt.shape}, value: {pt}")
            elif isinstance(points, torch.Tensor):
                print(f"  {name} shape: {points.shape}")

        print_point_shapes("Omega_K", Omega_K)
        print_point_shapes("Omega_L", Omega_L)
        print_point_shapes("Omega_I", Omega_I)

        # 合併三組點
        all_points = torch.cat([Omega_K, Omega_L, Omega_I], dim=0)
        print(f"合併總點數: {len(all_points)}")

        # 取出特徵並丟進 PointHead
        feats = model.point_refiner.pfe(feat, all_points)
        logits = model.point_refiner.ph(feats)
        pred = logits.sigmoid().squeeze().cpu().numpy()

        print(f"預測點數: {len(pred)}")
        print(f"預測機率範圍: min={pred.min():.4f}, max={pred.max():.4f}")
        

        import matplotlib.pyplot as plt

        img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        points_np = all_points.cpu().numpy()
        pred_np = pred  # shape: (768,)
        points_np = points_np[:, [1, 0]]  # 把 (i, j) → (x, y)

        plt.imshow(img_np)
        plt.scatter(points_np[:, 0], points_np[:, 1], c=pred_np, cmap='jet', s=10)
        plt.colorbar(label="Predicted probability")
        plt.title("PointHead predictions (Ω_K + Ω_L + Ω_I)")
        plt.show()


# python -m point_refiner.point_head

