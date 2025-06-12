import torch
import torch.nn as nn

class CoRE_Net(nn.Module):
    def __init__(self, encoder, decoder, point_refiner):
        super(CoRE_Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.point_refiner = point_refiner  # 包含 point head 與 feature extractor

    def forward(self, x):
        feats = self.encoder(x)                          # list of 4 feature maps
        deepest_feat = feats[-1]                         # [B, 516, 14, 14]

        pred_mask = self.decoder(feats)                  # [B, 1, 448, 448]

        logits, coords = self.point_refiner(
            pred_mask, deepest_feat         # pass to point refiner
        )

        return {
            "coarse_mask": pred_mask,        # 原 segmentation
            "point_logits": logits,          # 每個選中點的預測值 (raw logits)
            "point_coords": coords           # 每張圖的點座標
        }

