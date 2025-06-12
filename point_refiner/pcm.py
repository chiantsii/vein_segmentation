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
    def __init__(self, tau=4, top_l=128, top_i=128):
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

        kernel = torch.tensor([[1,1,1],[1,-10,1],[1,1,1]], device=coarse_mask.device, dtype=torch.float32).view(1,1,3,3)
        conv = F.conv2d(G_bin, kernel, padding=1)
        mask = (conv > self.tau).float()

        flat = mask.view(B, -1)
        _, idx = torch.topk(flat, self.top_l, dim=1)

        coords = []
        for b in range(B):
            ij = torch.stack([idx[b] // W, idx[b] % W], dim=1)
            coords.append(ij)
        return coords

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
        _, idx = torch.topk(flat, self.top_i, dim=1)

        coords = []
        for b in range(B):
            ij = torch.stack([idx[b] // W, idx[b] % W], dim=1)
            coords.append(ij)
        return coords

    def __call__(self, p_coarse):
        coarse_mask = (p_coarse > 0.5).float()
        Omega_L = self.get_breakpoints(coarse_mask)
        Omega_I = self.get_branching_points(coarse_mask)
        return Omega_L, Omega_I


if __name__ == '__main__':
    p_coarse = torch.rand(1, 1, 448, 448)
    cmm = ConfidenceMaskModule()
    pcm = PointCorrectionModule()
    A, A_unc, Omega_K = cmm(p_coarse)
    L_pts, I_pts = pcm(p_coarse)
    print(f"Breakpoints: {L_pts[0].shape}, Branching Points: {I_pts[0].shape}")
