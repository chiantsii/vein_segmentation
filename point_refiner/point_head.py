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
        logits = [self.mlp(x) for x in point_features]  # (K_i, 1)
        return logits


if __name__ == '__main__':
    dummy_points = [torch.randn(10, 256), torch.randn(5, 256)]
    ph = PointHead()
    out = ph(dummy_points)
    print(f"Batch 0 preds: {out[0].shape}, Batch 1 preds: {out[1].shape}")
