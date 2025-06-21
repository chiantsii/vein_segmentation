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
            A, Y_u = self.point_refiner(coarse_prob, high_feat)
        else:
            A = Y_u = None

        # ───────── 4. 封裝輸出 ─────────
        return {
            "coarse_logits": coarse_logits,   # ⇢ BCEWithLogitsLoss 用 還沒 sigmoid()
            "coarse_mask"  : coarse_prob,     # ⇢ 可視化 / CMM 門檻 有經過 sigmoid()

            # from refiner
            "confidence_mask": A,             # ⇢ A, 用來合成 pseudo label
            "Y_u"      : Y_u      # ⇢ M^r, 只在選點更新過的 mask

        }