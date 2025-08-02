import torch
import torch.nn.functional as F

class ConfidenceMaskModule:
    """
    CMM: Confidence Mask Module
    Selects high-confidence (>= Th) and low-confidence (<= Tl) pixels.
    Returns:
        A: binary mask of confident pixels
        A_unc: uncertainty map
        Omega_K: list of K most uncertain point indices
    """
    
    def __init__(self, T_high=0.9, T_low=0.1, top_k=512):
        self.T_high = T_high
        self.T_low = T_low
        self.top_k = top_k

    def get_confidence_mask(self, p_coarse):
        """
        Args:
            prob_map: Tensor (B, 1, H, W) - sigmoid output from decoder
        Returns:
            A: Tensor (B, 1, H, W) - binary mask where confident pixels are 1
        """
        A = ((p_coarse >= self.T_high) | (p_coarse <= self.T_low)).float()
        return A

    def get_uncertainty_map(self, p_coarse, A):
        """
        Args:
            prob_map: Tensor (B, 1, H, W)
            A: confident binary mask from get_confidence_mask
        Returns:
            A_unc: uncertainty score for each pixel (higher = more uncertain)
        """
        A_unc = torch.abs((p_coarse - 0.5) * (1.0 - A.float()))

        return A_unc

    def get_topk_uncertain_points(self, A_unc):
        """
        Args:
            A_unc: Tensor (B, 1, H, W)
        Returns:
            indices: list of (B, K, 2) - K most uncertain (i, j) per batch
        """
        B, C, H, W = A_unc.shape
        A_unc_flat = A_unc.view(B, -1)  # (B, H*W)
        _, topk_idx = torch.topk(A_unc_flat, self.top_k, dim=1)  # negative: smallest values
        if A_unc_flat.shape[1] < self.top_k:
            print(f"[CMM Warning] top_k={self.top_k} > total pixels={A_unc_flat.shape[1]}")

        indices = []
        for b in range(B):
            ij = torch.stack([topk_idx[b] // W, topk_idx[b] % W], dim=1)  # (K, 2)
            indices.append(ij)  # ✅ 你漏了這行！
        return indices if B > 1 else indices[0]  # ✅ 若只有一張圖直接 return Tensor (K, 2)

    def __call__(self, p_coarse):
        A = self.get_confidence_mask(p_coarse)
        A_unc = self.get_uncertainty_map(p_coarse, A)
        Omega_K = self.get_topk_uncertain_points(A_unc)
        return A, A_unc, Omega_K
    

# ---------------- TEST FUNCTION (inside __main__) ----------------
def test_cmm():
    from core_net import CoRE_Net
    from encoder import Encoder
    from decoder import Decoder
    from point_refiner.point_refiner import PointRefiner

    # 參數設定
    image_path = r"data/LVD2021/36_Holly_labels/train/36_2/2_0003_背景.jpg"
    model_path = "checkpoints/36_Holly_labels/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = CoRE_Net(Encoder(), Decoder(), PointRefiner())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # 圖片處理
    from torchvision import transforms
    from PIL import Image

    # 圖片處理
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 預測 coarse mask
    with torch.no_grad():
        output = model(img_tensor, use_refiner=False)
        coarse_sigmoid = output["coarse_mask"]

    # CMM 處理
    cmm = ConfidenceMaskModule(top_k=512)
    A, A_unc, Omega_K = cmm(coarse_sigmoid)

    # 可視化
    import matplotlib.pyplot as plt
    coarse_np = coarse_sigmoid.squeeze().cpu().numpy()
    unc_map = A_unc.squeeze().cpu().numpy()
    topk_points = Omega_K[0].cpu().numpy()

    plt.imshow(coarse_np, cmap='gray')
    plt.title("Coarse Mask")
    plt.axis("off")
    plt.show()

    plt.imshow(unc_map, cmap='hot')
    plt.title("Uncertainty Map (A_unc)")
    plt.axis("off")
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(unc_map, cmap='hot')
    ax.scatter(topk_points[:, 1], topk_points[:, 0], s=10, c='cyan', label='Top-K Points')
    ax.set_title("Top-K Uncertain Points Overlay")
    ax.axis("off")
    plt.legend()
    plt.show()

    print("✅ 顯示完成。")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    test_cmm()


# python -m point_refiner.cmm
