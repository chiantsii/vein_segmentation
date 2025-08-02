import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import VeinSegmentationDataset
from pseudoloader import PseudoLabelDataset
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from core_net import CoRE_Net
from infer_labels import infer_labels
import torch.nn.functional as F

# dice loss + iou loss
def dice_iou_loss(y_pred, y_true, smooth=1e-6):
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)
    intersection = (y_pred_flat * y_true_flat).sum(1)
    union = y_pred_flat.sum(1) + y_true_flat.sum(1)
    dice = (2 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    dice_loss = 1 - dice.mean()
    iou_loss  = 1 - iou.mean()
    return dice_loss, iou_loss

# ---------------- 參數 ---------------- #
DATA_ROOT      = "data/LVD2021/36_Holly_labels/train"
VALID_ROOT     = "data/LVD2021/36_Holly_labels/valid"
CKPT_DIR       = "checkpoints/36_Holly_labels"
CKPT_OUT       = os.path.join(CKPT_DIR, "selftrain_model.pth")
BASE_PSEUDO    = "data/LVD2021/36_Holly_labels/pseudo"
PRETRAIN_CKPT  = os.path.join(CKPT_DIR, "best_model.pth")
OUTER_EPOCHS   = 10
INNER_EPOCHS   = 10
BATCH_SIZE     = 8
LR             = 0.001
PATIENCE       = 5
POINT_LOSS_WEIGHT = 1.0
# -------------------------------------- #

import csv

log_file = os.path.join(CKPT_DIR, "loss_log.csv")
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["outer_epoch", "inner_epoch", "train_loss", "coarse_loss", "point_loss", "dice_loss", "iou_loss"])

def build_model(device):
    model = CoRE_Net(Encoder(), Decoder(), PointRefiner()).to(device)
    return model

def train_self_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for outer_epoch in range(OUTER_EPOCHS):
        print(f"\n====== Outer Epoch {outer_epoch} ======")

        model = build_model(device)
        ckpt_path = PRETRAIN_CKPT if outer_epoch == 0 else os.path.join(CKPT_DIR, "train.th")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"📌 使用模型權重: {ckpt_path}")
        print(f"🔁 使用 PointRefiner: ✅ 開啟")

        dataset = VeinSegmentationDataset(DATA_ROOT, image_size=(448, 448))
        pseudo_root = os.path.join(BASE_PSEUDO, f"epoch_{outer_epoch}")
        infer_labels(model, dataset, device, save_dir=pseudo_root)

        train_dataset = PseudoLabelDataset(pseudo_root)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_dataset = VeinSegmentationDataset(VALID_ROOT, image_size=(448, 448))
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4)

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        best_val_loss = float('inf')
        patience_counter = 0
        best_train_loss = float('inf')

        for inner_epoch in range(INNER_EPOCHS):
            print(f"Outer {outer_epoch} | Inner {inner_epoch}")
            train_loss = 0.0

            for batch_idx, batch in enumerate(tqdm(train_loader)):
                img = batch["image"].to(device)
                y_u = batch["pseudo_mask"].to(device)
                b_map = batch["b_map"].to(device)

                out = model(img)
                pred_logits = out["coarse_logits"]
                pred_point = out["point_logits"]
                point_coords = out["point_coords"]


                # ① coarse loss
                coarse_loss_map = nn.functional.binary_cross_entropy_with_logits(pred_logits, y_u, reduction="none")
                coarse_loss = (coarse_loss_map * b_map).sum() / (b_map.sum() + 1e-6)

                # ② point-wise loss
                point_loss = 0.0
                valid_count = 0

                for i in range(len(pred_point)):
                    if pred_point[i] is None or point_coords[i] is None:
                        continue

                    logits = pred_point[i].squeeze(1)  # shape: (K,)
                    coords = point_coords[i].float()  # shape: (K, 2)
                    mask_i = y_u[i, 0]  # pseudo label: (H, W)

                    if logits.shape[0] == 0:
                        continue

                    H, W = mask_i.shape

                    logits = pred_point[i].squeeze(1)  # (K,)
                    coords = point_coords[i].long()    # (K, 2)
                    x = coords[:, 0].clamp(0, W-1)
                    y = coords[:, 1].clamp(0, H-1)

                    gt = mask_i[y, x]  # shape: (K,)

                    # --- compute loss ---
                    point_loss += F.binary_cross_entropy_with_logits(logits, gt)
                    valid_count += 1 

                if valid_count > 0:
                    point_loss = point_loss / valid_count

                # ③ dice + iou loss on refined
                dice_loss, iou_loss = dice_iou_loss(torch.sigmoid(pred_logits), y_u)
                dice_iou = dice_loss + iou_loss

                total_loss = coarse_loss + point_loss + 0.5*dice_iou
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

                if batch_idx == 0:
                    vis_dir = f"debug_vis/train_outer{outer_epoch}_inner{inner_epoch}"
                    os.makedirs(vis_dir, exist_ok=True)
                    print(f"[Loss] Coarse: {coarse_loss.item():.4f} | Point: {point_loss.item():.4f} | Dice+IoU: {dice_iou.item():.4f}")
                    save_image(torch.sigmoid(pred_logits[0]), os.path.join(vis_dir, "coarse.png"))
                    save_image(y_u[0], os.path.join(vis_dir, "yu.png"))
                    save_image(img[0].cpu(), os.path.join(vis_dir, "input.png"))

            train_loss /= len(train_loader)
            print(f"Train Loss: {train_loss:.4f}")

            # 🔽 Save best train model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "train.th"))
                print("✅ Save best TRAIN model")

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    outer_epoch, inner_epoch, 
                    round(train_loss, 6), 
                    round(coarse_loss.item(), 6), 
                    round(point_loss.item() if isinstance(point_loss, torch.Tensor) else point_loss, 6), 
                    round(dice_loss.item(), 6), 
                    round(iou_loss.item(), 6)
                ])


if __name__ == "__main__":
    train_self_training()

# python self_train.py