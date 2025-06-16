import torch
import torch.nn as nn

class CoRE_Net(nn.Module):
    def __init__(self, encoder, decoder, point_refiner):
        super(CoRE_Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.point_refiner = point_refiner  # 包含 point head 與 feature extractor

    def forward(self, x, use_refiner=True):
        feats = self.encoder(x)
        deepest_feat = feats[-1]
        pred_mask = self.decoder(feats)

        if use_refiner:
            logits, point_coords, point_mask, refined_mask, Yu = self.point_refiner(pred_mask, deepest_feat)
        else:
            logits, point_coords, point_mask, refined_mask, Yu = None, None, None, None, None

        return {
            "coarse_mask": pred_mask,
            "refined_mask": refined_mask,
            "point_mask": point_mask,
            "point_logits": logits,
            "point_coords": point_coords,
            "pseudo_label": Yu  # 這就是 Y_u
        }



