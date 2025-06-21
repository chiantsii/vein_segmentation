import os, torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import VeinSegmentationDataset
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from core_net import CoRE_Net

# ---------------- 使用者可改參數 ---------------- #
DATA_ROOT       = "data/LVD2021/36_Holly_labels/train"
PRETRAIN_CKPT   = "checkpoints/36_Holly_labels/best_model.pth"
CKPT_OUT        = "checkpoints/self_train.pth"
EPOCHS          = 100
BATCH_SIZE      = 5
LR              = 1e-4
# ------------------------------------------------ #

def train_self_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CoRE_Net(Encoder(), Decoder(), PointRefiner()).to(device)

    if os.path.exists(PRETRAIN_CKPT):
        model.load_state_dict(torch.load(PRETRAIN_CKPT, map_location=device))
        print(f"📥 Loaded pretrained coarse weights: {PRETRAIN_CKPT}")
    else:
        print("⚠️ 找不到預訓練權重，將從頭訓練。")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    dataset   = VeinSegmentationDataset(DATA_ROOT)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    pixel_criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        save_dir = f"predictions/self_train/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        saved_img = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            imgs = batch["image"].to(device)

            # === forward，這裡要開啟 Point Refiner =========
            out = model(imgs, use_refiner=True)

            coarse_logits = out["coarse_logits"]       # (B,1,H,W) logits
            coarse_prob   = out["coarse_mask"]         # sigmoid(coarse_logits)
            A             = out["confidence_mask"]     # (B,1,H,W)
            Y_u     = out["Y_u"]           # (B,1,H,W)


            # ---------- 損失計算 ----------
            pix_loss_map = pixel_criterion(coarse_logits, Y_u)
            pix_loss     = pix_loss_map.mean()
            loss         = pix_loss  # + pt_loss 若有 point loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # ---------- 可視化 ----------
            if saved_img < 1:
                torchvision.utils.save_image(coarse_prob[0], f"{save_dir}/sample_coarse.png")
                torchvision.utils.save_image(Y_u[0],     f"{save_dir}/sample_refined.png")
                torchvision.utils.save_image(A[0],             f"{save_dir}/sample_A_mask.png")
                saved_img += 1

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch} | avg_loss = {avg:.4f}")
        print("coarse min/max:", coarse_prob.min().item(), coarse_prob.max().item())


        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"✅ Saved new best (loss={best_loss:.4f}) → {CKPT_OUT}")

if __name__ == "__main__":
    train_self_training()


# python self_train.py