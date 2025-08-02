import torch
import torch.nn as nn


class CoRE_Net(nn.Module):
    def __init__(self, encoder, decoder, point_refiner):
        super().__init__()
        self.encoder       = encoder
        self.decoder       = decoder
        self.point_refiner = point_refiner        # CMM + PCM + PFE + PH

    def forward(self, x, use_refiner=True):

        # ───────── 1. backbone ─────────
        encoder_feats = self.encoder(x)           # list; 最深層放最後
        high_feat     = encoder_feats[-1]         # DAC+RMP 輸出 (B,C,h,w)

        # ───────── 2. decoder ──────────
        coarse_logits = self.decoder(encoder_feats)      # (B,1,H,W) ＝ logits
        coarse_prob   = torch.sigmoid(coarse_logits)     # (B,1,H,W) ＝ 0–1

        # ───────── 3. point refiner ────
        if use_refiner:
            A, Y_u, point_coords, point_logits = self.point_refiner(coarse_prob, high_feat)

            # 將修正點 logits 融合進 coarse mask → refined mask
            refined_mask = coarse_prob.clone()  # (B,1,H,W)

            for i in range(len(point_coords)):
                if point_coords[i] is None or point_logits[i] is None:
                    continue
                coords = point_coords[i].long()  # (K, 2), usually (x, y)
                refined_mask[i, 0, coords[:, 0], coords[:, 1]] = torch.sigmoid(point_logits[i].squeeze(1))

        else:
            A = Y_u = point_coords = point_logits = None
            refined_mask = coarse_prob  # 沒有 refiner，直接用 coarse mask



        # ───────── 4. 封裝輸出 ─────────
        return {
            "features"        : high_feat,
            "coarse_logits"   : coarse_logits,
            "coarse_mask"     : coarse_prob,
            "confidence_mask" : A,
            "Y_u"    : Y_u,
            "point_coords"    : point_coords,
            "point_logits"    : point_logits,
            "refined_mask"    : refined_mask 
        }

