import torch
import torch.nn as nn
from torchvision.ops import roi_align

class PointFeatureExtractor:
    """
    PFE: Point Feature Extractor
    - 使用 RoIAlign 對 coarse features 抽取每個選中點的特徵向量
    - 支援多張圖（batch），每張圖多個點
    """
    def __init__(self, output_size=1):
        self.output_size = output_size  # RoIAlign 輸出大小 (1x1)

    def __call__(self, feature_map, point_coords): 
        """
        Args:
            feature_map: Tensor (B, C, H, W) - 來自 encoder 的語意特徵
            point_coords: list of Tensors (K_i, 2) for each image, 每點是 (i, j)
        Returns:
            point_features: list of Tensors (K_i, C) - 每點對應的特徵向量
        """
        B, C, H, W = feature_map.shape
        all_rois = []
        batch_idx = []

        for b, coords in enumerate(point_coords):
            if coords.shape[0] == 0:
                continue
            for pt in coords:
                i, j = pt[0].item(), pt[1].item()
                # 建立一個超小 box centered at (j, i) in feature map 空間
                x1 = x2 = j / W * feature_map.shape[3]
                y1 = y2 = i / H * feature_map.shape[2]
                all_rois.append(torch.tensor([b, x1, y1, x2+1e-4, y2+1e-4], device=feature_map.device))
                batch_idx.append(b)

        if len(all_rois) == 0:
            return [torch.empty((0, C), device=feature_map.device) for _ in range(B)]

        rois = torch.stack(all_rois, dim=0)  # (N_all, 5)
        pooled = roi_align(feature_map, rois, output_size=self.output_size, aligned=True)  # (N_all, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (N_all, C)

        # 將 pooled 特徵依照 batch 拆回去
        pointer = 0
        outputs = [[] for _ in range(B)]
        for b, coords in enumerate(point_coords):
            n = coords.shape[0]
            outputs[b] = pooled[pointer:pointer+n]
            pointer += n

        return outputs


if __name__ == '__main__':
    feat = torch.randn(2, 256, 14, 14)
    coords = [torch.randint(0, 14, (10, 2)), torch.randint(0, 14, (5, 2))]
    pfe = PointFeatureExtractor()
    out = pfe(feat, coords)
    print(f"Batch 0 feature: {out[0].shape}, Batch 1 feature: {out[1].shape}")
