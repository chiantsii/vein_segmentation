import torch
import torch.nn as nn
from point_refiner.cmm import ConfidenceMaskModule
from point_refiner.pcm import PointCorrectionModule
from point_refiner.point_features import PointFeatureExtractor
from point_refiner.point_head import PointHead

class PointRefiner(nn.Module):
    """
    將 CMM + PCM + PFE + PH 整合為一個模組：
    - Input: decoder 的 coarse mask 與 encoder 的 high-level feature
    - Output: 所有選中的點的 refined logits 或 binary label
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
            prob_map: (B, 1, H, W) - decoder output after sigmoid
            coarse_mask: (B, 1, H, W) - binary coarse mask (Y_coarse)
            feature_map: (B, C, h, w) - from encoder, after DAC+RMP
        Returns:
            logits: list of (K_i, 1) - each point's raw prediction logit
            point_coords: list of (K_i, 2) - used for re-projection later
        """

        A, A_unc, Omega_K = self.cmm(p_coarse_mask)          # Uncertain points
        Omega_L, Omega_I = self.pcm(p_coarse_mask)        # Breakpoints + Branching points

        point_coords = []
        for b in range(len(Omega_K)):
            all_pts = torch.cat([Omega_K[b], Omega_L[b], Omega_I[b]], dim=0)
            point_coords.append(all_pts)

        point_feats = self.pfe(feature_map, point_coords)  # List[Tensor(K, C)]
        logits = self.ph(point_feats)                      # List[Tensor(K, 1)]
        return logits, point_coords
