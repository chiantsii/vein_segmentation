# test_pfe.py
import torch
from encoder import Encoder
from point_refiner.point_features import PointFeatureExtractor

if __name__ == '__main__':
    x = torch.randn(2, 3, 448, 448)
    encoder = Encoder()
    features = encoder(x)
    feat = features[-1]  # [2, 516, 14, 14]
    coords = [torch.randint(0, 14, (10, 2)), torch.randint(0, 14, (5, 2))]
    pfe = PointFeatureExtractor()
    out = pfe(feat, coords)
    print(f"Batch 0 feature: {out[0].shape}, Batch 1 feature: {out[1].shape}")
