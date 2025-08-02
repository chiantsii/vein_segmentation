import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import VeinSegmentationDataset
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from core_net import CoRE_Net

# ---------------- 使用者參數 ---------------- #
VALID_ROOT     = "data/LVD2021/36_Holly_labels/all"
CKPT_PATH      = "checkpoints/36_Holly_labels/train.th"
IMAGE_SIZE     = (448, 448)
BATCH_SIZE     = 1
# ------------------------------------------- #

def build_model(device):
    model = CoRE_Net(Encoder(), Decoder(), PointRefiner()).to(device)
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    total_inter = 0
    total_union = 0
    total_dice = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            img = batch["image"].to(device)
            gt_mask = batch["gt_mask"].to(device)  # (B, 1, H, W)

            out = model(img)
            coarse_logits = out["coarse_logits"]           # (B, 1, H, W)
            point_logits = out["point_logits"]             # list[B] 每張圖部分點的 refined logits
            point_coords = out["point_coords"]             # list[B] 每張圖 refined 的點座標

            prob_map = torch.sigmoid(coarse_logits).clone()  # base 是 coarse 預測

            for i in range(len(point_logits)):
                if point_logits[i] is None or point_coords[i] is None:
                    continue
                coords = point_coords[i].long()  # shape: (K, 2)
                logits = point_logits[i].sigmoid().view(-1)  # (K,) → sigmoid 後為 prob 值

                x = coords[:, 0].clamp(0, prob_map.shape[-1] - 1)
                y = coords[:, 1].clamp(0, prob_map.shape[-2] - 1)
                prob_map[i, 0, y, x] = logits  # 🔁 更新 coarse map 的部分區域

            pred_bin = (prob_map > 0.5).float()

            # Accuracy
            correct = (pred_bin == gt_mask).float().sum()
            total_pixels += gt_mask.numel()
            correct_pixels += correct.item()

            # IoU & Dice
            inter = (pred_bin * gt_mask).sum(dim=(1, 2, 3))
            union = ((pred_bin + gt_mask) >= 1).float().sum(dim=(1, 2, 3))
            dice = (2 * inter) / (pred_bin.sum(dim=(1,2,3)) + gt_mask.sum(dim=(1,2,3)) + 1e-6)

            total_inter += inter.sum().item()
            total_union += union.sum().item()
            total_dice += dice.sum().item()

    accuracy = correct_pixels / total_pixels
    iou = total_inter / (total_union + 1e-6)
    dice = total_dice / len(dataloader.dataset)

    print(f"\n✅ Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"IoU     : {iou:.4f}")
    print(f"Dice    : {dice:.4f}")

    return accuracy, iou, dice


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    model = build_model(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    # 驗證集 DataLoader
    val_dataset = VeinSegmentationDataset(VALID_ROOT, image_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # 執行評估
    evaluate_model(model, val_loader, device)

# python test.py