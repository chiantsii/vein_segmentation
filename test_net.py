import torch
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from core_net import CoRE_Net

if __name__ == "__main__":
    x = torch.randn(2, 3, 448, 448)  # 模擬兩張圖

    encoder = Encoder()
    decoder = Decoder()
    refiner = PointRefiner(top_k=768, top_l=128, top_i=128)
    model = CoRE_Net(encoder, decoder, refiner)

    output = model(x)

    print("Coarse mask:", output["coarse_mask"].shape)        # [2, 1, 448, 448]
    print("Logits[0]:", output["point_logits"][0].shape)      # [1024, 1] or whatever topk+topl+topi
    print("Coords[0]:", output["point_coords"][0].shape)      # [1024, 2]

