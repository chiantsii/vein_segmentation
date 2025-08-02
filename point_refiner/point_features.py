import torch
import torch.nn as nn
from torchvision.ops import roi_align

class PointFeatureExtractor:
    """
    PFE: Point Feature Extractor
    - 使用 RoIAlign 對 coarse features 抽取每個選中點的特徵向量
    - 同時支援單張圖 (Tensor) 或多圖 batch (List[Tensor])
    """
    def __init__(self, output_size=1):
        self.output_size = output_size  # RoIAlign 輸出大小 (1x1)

    def __call__(self, feature_map, point_coords):
        """
        Args:
            feature_map: Tensor (B, C, H, W)
            point_coords:
                - List[Tensor(K_i, 2)]，每張圖多點
                - 或單一 Tensor(K, 2)，表示只有一張圖
        Returns:
            List[Tensor(K_i, C)]，每張圖的點特徵
        """
        B, C, H, W = feature_map.shape

        # ✅ 若為單圖情況，自動包成 list
        if isinstance(point_coords, torch.Tensor):
            point_coords = [point_coords]
            is_single = True
        else:
            is_single = False

        all_rois = []
        roi_batch_ids = []

        for b, coords in enumerate(point_coords):
            if coords.numel() == 0:
                continue
            for pt in coords:
                i, j = pt[0].item(), pt[1].item()  # (y, x)
                x1 = x2 = j / (W - 1) * W
                y1 = y2 = i / (H - 1) * H
                all_rois.append(torch.tensor([b, x1, y1, x2 + 1e-4, y2 + 1e-4], device=feature_map.device))
                roi_batch_ids.append(b)

        if len(all_rois) == 0:
            return [torch.empty((0, C), device=feature_map.device) for _ in range(B)]

        rois = torch.stack(all_rois, dim=0)  # (N, 5)
        pooled = roi_align(feature_map, rois, output_size=self.output_size, aligned=True)  # (N, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (N, C)

        # 分回每張圖
        outputs = [[] for _ in range(B)]
        pointer = 0
        for b, coords in enumerate(point_coords):
            k = coords.shape[0]
            outputs[b] = pooled[pointer:pointer + k]
            pointer += k

        return outputs[0] if is_single else outputs


if __name__ == '__main__':
    feat = torch.randn(2, 256, 14, 14)
    coords = [torch.randint(0, 14, (10, 2)), torch.randint(0, 14, (5, 2))]
    pfe = PointFeatureExtractor()
    out = pfe(feat, coords)
    print(f"Batch 0 feature: {out[0].shape}, Batch 1 feature: {out[1].shape}")
