import torch
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner


if __name__ == '__main__':
    # 模擬輸入圖像
    x = torch.randn(2, 3, 448, 448)

    # 模型初始化
    encoder = Encoder()
    decoder = Decoder()
    refiner = PointRefiner(top_k=768, top_l=128, top_i=128)

    # 特徵抽取與 coarse segmentation
    feats = encoder(x)
    p_coarse_mask = decoder(feats)  # shape: [2, 1, 448, 448]

    # Encoder 最深層特徵
    deepest_feat = feats[-1]  # [2, 516, 14, 14]

    # 執行 refinement
    logits, coords = refiner(p_coarse_mask,  deepest_feat)

    # 輸出結果確認
    for i in range(len(logits)):
        print(f"Image {i}:")
        print(f"  Point logits shape: {logits[i].shape}")   # [K_i, 1]
        print(f"  Point coords shape: {coords[i].shape}")   # [K_i, 2]")
