import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
from point_refiner.cmm import ConfidenceMaskModule

class PointCorrectionModule:
    """
    PCM: Point Correction Module
    - Find breakpoints using gradient magnitude + local neighbor count
    - Find branching points using divergence of gradient field
    """
    def __init__(self, tau=0, top_l=128, top_i=128):
        self.tau = tau      # threshold for breakpoints
        self.top_l = top_l  # number of breakpoints
        self.top_i = top_i  # number of branching points

    def compute_gradient(self, mask):
        """
        Compute gradient using Sobel operator
        Args:
            mask: Tensor (B, 1, H, W) - binary coarse mask
        Returns:
            Gx, Gy, GA: gradient x, y, and magnitude
        """
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        Gx = F.conv2d(mask, sobel_x, padding=1)
        Gy = F.conv2d(mask, sobel_y, padding=1)
        GA = torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-8)
        return Gx, Gy, GA

    def get_breakpoints(self, coarse_mask):
        """
        Find breakpoints: where local gradient exists but has few vein neighbors
        Returns:
            list of (B, L, 2) - top L breakpoint coordinates per image
        """
        B, _, H, W = coarse_mask.shape
        _, _, GA = self.compute_gradient(coarse_mask)
        G_bin = (GA > 0).float()  # edge indicator
        G_bin = G_bin * (coarse_mask > 0.5).float()

        kernel = torch.tensor([[1,1,1],[1,-10,1],[1,1,1]], device=coarse_mask.device, dtype=torch.float32).view(1,1,3,3)
        conv = F.conv2d(G_bin, kernel, padding=1)
        mask = (conv > 2).float()

        flat = mask.view(B, -1) 
        _, idx = torch.topk(flat, self.top_l, dim=1, largest=True)

        coords = []
        for b in range(B):
            ij = torch.stack([idx[b] // W, idx[b] % W], dim=1)
            coords.append(ij)
        return coords if B > 1 else coords[0]

    def get_branching_points(self, coarse_mask):
        """
        Find branching points: where divergence of gradient is highest
        Returns:
            list of (B, I, 2) - top I branching point coordinates per image
        """
        B, _, H, W = coarse_mask.shape
        Gx, Gy, _ = self.compute_gradient(coarse_mask)

        dx = Gx[:, :, :, 2:] - Gx[:, :, :, :-2]  # central difference
        dy = Gy[:, :, 2:, :] - Gy[:, :, :-2, :]

        dx = F.pad(dx, (1,1,0,0))
        dy = F.pad(dy, (0,0,1,1))

        divergence = torch.abs(dx + dy)  # (B, 1, H, W)
        flat = divergence.view(B, -1)
        _, idx = torch.topk(flat, self.top_i, dim=1, largest=True)

        coords = []
        for b in range(B):
            ij = torch.stack([idx[b] // W, idx[b] % W], dim=1)
            coords.append(ij)
        return coords if B > 1 else coords[0]


    def __call__(self, p_coarse):
        coarse_mask = (p_coarse > 0.5).float()
        Omega_L = self.get_breakpoints(coarse_mask)
        Omega_I = self.get_branching_points(coarse_mask)
        return Omega_L, Omega_I
    
# ---------------- TEST FUNCTION (inside __main__) ----------------
def test_pcm():
    from core_net import CoRE_Net
    from encoder import Encoder
    from decoder import Decoder
    from point_refiner.point_refiner import PointRefiner 
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    image_path = r"data/LVD2021/36_Holly_labels/train/36_2/2_0003_背景.jpg"
    model_path = "checkpoints/36_Holly_labels/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = CoRE_Net(Encoder(), Decoder(),  PointRefiner())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval().to(device)

    # 圖片預處理
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

    # 使用 PCM
    pcm = PointCorrectionModule()
    Omega_L, Omega_I = pcm(coarse_sigmoid)

    # 可視化 coarse mask + PCM points
    coarse_np = coarse_sigmoid.squeeze().cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(coarse_np, cmap='gray')

    if Omega_L:
        pts_l = Omega_L[0].cpu().numpy()
        ax.scatter(pts_l[:, 1], pts_l[:, 0], s=10, c='blue', label='Breakpoints (ΩL)')
    if Omega_I:
        pts_i = Omega_I[0].cpu().numpy()
        ax.scatter(pts_i[:, 1], pts_i[:, 0], s=10, c='red', label='Branch Points (ΩI)')

    ax.set_title("PCM Points on Coarse Mask")
    ax.axis("off")
    ax.legend()
    plt.show()

    print("✅ PCM 測試完成。")


if __name__ == "__main__":
    test_pcm()

# python -m point_refiner.pcm
