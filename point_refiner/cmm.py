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
        A_unc = torch.abs((1.0 - A.float()) * (p_coarse - 0.5))

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
        _, topk_idx = torch.topk(-A_unc_flat, self.top_k, dim=1)  # negative: smallest values

        indices = []
        for b in range(B):
            ij = torch.stack([topk_idx[b] // W, topk_idx[b] % W], dim=1)  # (K, 2)
            indices.append(ij)
        return indices

    def __call__(self, p_coarse):
        A = self.get_confidence_mask(p_coarse)
        A_unc = self.get_uncertainty_map(p_coarse, A)
        Omega_K = self.get_topk_uncertain_points(A_unc)
        return A, A_unc, Omega_K


if __name__ == "__main__":
    p_coarse = torch.rand(1, 1, 448, 448)
    cmm = ConfidenceMaskModule()
    A, A_unc, Omega_K = cmm(p_coarse)
    print(f"Confident Mask A: {A.shape}, Uncertainty Map: {A_unc.shape}, Top-K Points: {Omega_K[0].shape}")

# python cmm.py