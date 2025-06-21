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
    def __init__(self, top_k=512, top_l=128, top_i=128):
        super().__init__()
        self.cmm = ConfidenceMaskModule(top_k=top_k)
        self.pcm = PointCorrectionModule(top_l=top_l, top_i=top_i)
        self.pfe = PointFeatureExtractor()
        self.ph  = PointHead()

    def forward(self, p_coarse_mask, feature_map):
        B, _, H, W = p_coarse_mask.shape

        # 1) confidence mask A  &  抽樣點 Ω
        A, A_unc, Omega_K = self.cmm(p_coarse_mask)
        Omega_L, Omega_I  = self.pcm(p_coarse_mask)

        # 2) 合併座標
        point_coords = [torch.cat([Omega_K[b], Omega_L[b], Omega_I[b]], 0)
                        for b in range(B)]

        # 3) Point Head → logits
        point_feats = self.pfe(feature_map, point_coords)   # list[(K_i,C)]
        logits_list = self.ph(point_feats)                  # list[(K_i,1)]

        # 4) 組 Y_refined：只在 Ω 寫入 σ(logit)
        Y_refined = torch.zeros_like(p_coarse_mask)         # 其餘位置 0
        for b in range(B):
            if point_coords[b].numel() == 0:
                continue
            coords = point_coords[b].long()
            y = coords[:, 0].clamp(0, H - 1)
            x = coords[:, 1].clamp(0, W - 1)
            prob = torch.sigmoid(logits_list[b].squeeze(1))
            Y_refined[b, 0, y, x] = prob

        # 5) Yu = A·p  +  Y_refined     （不乘 1-A）
        Y_u = (A * p_coarse_mask + Y_refined).detach()

        return A, Y_u
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from core_net import CoRE_Net
    from encoder import Encoder
    from decoder import Decoder

    # ==== 1. 參數設定 ====
    image_path = "/Users/chiantsii/Desktop/vein_seg/data/LVD2021/36_Holly_labels/test/36_22/22_0003_背景.jpg"  # 請修改為你圖片的路徑
    ckpt_path  = "checkpoints/36_Holly_labels/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    with torch.no_grad():
        feats = model.encoder(img_tensor)
        high_feat = feats[-1]
        coarse_logits = model.decoder(feats)
        coarse_prob = torch.sigmoid(coarse_logits)

        A, Y_u = model.point_refiner(coarse_prob, high_feat)

    # ==== 4. 顯示結果 ====
    def show_tensor(t, title):
        img = t.squeeze().cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    show_tensor(coarse_prob, "Coarse Mask")
    show_tensor(A,           "Confidence Mask A")
    show_tensor(Y_u,         "Pseudo Label Y_u")


# python -m point_refiner.point_refiner