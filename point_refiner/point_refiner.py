import torch
import torch.nn as nn
from .cmm import ConfidenceMaskModule
from .pcm import PointCorrectionModule
from .point_features import PointFeatureExtractor
from .point_head import PointHead
from PIL import Image
import torchvision.transforms as T
import numpy as np

import torch
import torch.nn as nn
from .cmm import ConfidenceMaskModule
from .pcm import PointCorrectionModule
from .point_features import PointFeatureExtractor
from .point_head import PointHead
from PIL import Image
import torchvision.transforms as T
import numpy as np

class PointRefiner(nn.Module):  # ✅ 你漏了這一段
    def __init__(self, top_k=512, top_l=128, top_i=128):
        super().__init__()
        self.cmm = ConfidenceMaskModule(top_k=top_k)
        self.pcm = PointCorrectionModule(top_l=top_l, top_i=top_i)
        self.pfe = PointFeatureExtractor()
        self.ph  = PointHead()

    def forward(self, p_coarse_mask, feature_map):
        B, _, H, W = p_coarse_mask.shape

        # 1) 信心區 A + 三組抽樣點 Ω
        A, A_unc, Omega_K = self.cmm(p_coarse_mask)
        Omega_L, Omega_I  = self.pcm(p_coarse_mask)

        # ✅ 包裝成 List[Tensor] 格式（支援單圖或 batch）
        if isinstance(Omega_K, torch.Tensor): Omega_K = [Omega_K]
        if isinstance(Omega_L, torch.Tensor): Omega_L = [Omega_L]
        if isinstance(Omega_I, torch.Tensor): Omega_I = [Omega_I]

        # 2) 合併座標
        point_coords = [torch.cat([ok, ol, oi], dim=0)
                        for ok, ol, oi in zip(Omega_K, Omega_L, Omega_I)]

        # 3) Point Head → logits
        point_feats = self.pfe(feature_map, point_coords)
        logits_list = [self.ph(feats) for feats in point_feats]

        # 4) 建立 Y_refined（僅對點位置更新）
        Y_refined = torch.zeros_like(p_coarse_mask)
        for b in range(B):
            if point_coords[b].numel() == 0:
                continue
            coords = point_coords[b].long()
            y = coords[:, 0].clamp(0, H - 1)
            x = coords[:, 1].clamp(0, W - 1)
            prob = torch.sigmoid(logits_list[b].squeeze(1))
            Y_refined[b, 0, y, x] = prob

        # 5) 組合 Y_u：高信心區保留 coarse，低信心區用 refined
        Y_coarse_bin = (Y_refined > 0.5).float()
        Y_u = (A * p_coarse_mask + (1 - A) * Y_refined).detach()

        return A, Y_u, point_coords, logits_list




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from core_net import CoRE_Net
    from encoder import Encoder
    from decoder import Decoder

    # ==== 1. 參數設定 ====
    image_path = r"data/LVD2021/36_Holly_labels/train/36_2/2_0003_背景.jpg"
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

        A, Y_u, coords, logits = model.point_refiner(coarse_prob, high_feat)


    # ==== 4. 顯示結果 ====
    def show_tensor(t, title):
        img = t.squeeze().cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    print(f"\n🧪 Batch Size: {len(coords)}")
    for i, c in enumerate(coords):
        print(f"Batch {i}: coords.shape = {c.shape}")
        print(f"Sample coords (前 5 個):\n{c[:5].cpu().numpy()}")


    show_tensor(coarse_prob, "Coarse Mask")
    show_tensor(A,           "Confidence Mask A")
    show_tensor(Y_u,         "Pseudo Label Y_u")


# python -m point_refiner.point_refiner