# pretrain.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from core_net import CoRE_Net
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from dataloader import VeinSegmentationDataset

# dice loss + iou loss
def dice_iou_loss(y_pred, y_true, smooth=1e-6):
    """
    y_pred: sigmoid 後的預測 (B, 1, H, W)    # 0-1
    y_true: ground truth mask (B, 1, H, W)  # 0/1
    """
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    intersection = (y_pred_flat * y_true_flat).sum(1)
    union = y_pred_flat.sum(1) + y_true_flat.sum(1)

    dice = (2 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)

    dice_loss = 1 - dice.mean()
    iou_loss  = 1 - iou.mean()

    return dice_loss, iou_loss

# -------------------------- 參數 --------------------------
which       = '36_Holly_labels'
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 0.001
CKPT_DIR    = "checkpoints/" + which
PRED_DIR    = "predictions/" + which
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT   = "data/LVD2021/36_Holly_labels/pretrain"
# ----------------------------------------------------------

def train():
    # 1. 建立模型 ---------------------------------------------------------
    model = CoRE_Net(Encoder(), Decoder(), PointRefiner()).to(DEVICE)

    # 鎖定 point refiner
    for p in model.point_refiner.parameters():
        p.requires_grad = False

    # 優化 encoder + decoder
    # 只訓練 encoder 和 decoder 的參數
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(params, lr=LR)

    # BCE loss 強化葉脈參數：5
    # pos_weight = torch.tensor([5.0]).to(DEVICE)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()

    # 2. 資料 -------------------------------------------------------------
    dataset   = VeinSegmentationDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    best_loss = float("inf")
    
    # csv 記錄 avg loss / max / min
    log_path = os.path.join(CKPT_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,total_loss,bce_loss,dice_loss,iou_loss,pred_min,pred_max\n")

    # 3. 訓練迴圈 ---------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=80):
            imgs        = batch["image"].to(DEVICE)            # [B,3,H,W] 0–1
            gt_masks    = batch["gt_mask"].to(DEVICE)          # [B,1,H,W] 0/1
            leaf_masks  = batch["leaf_mask"].to(DEVICE).float() # [B,1,H,W] 0/1

            # forward（關閉 refiner）
            coarse_logits = model(imgs, use_refiner=False)["coarse_logits"]  # logits
            coarse_pre = torch.sigmoid(coarse_logits)

            # per-pixel BCE
            bce_loss = criterion(coarse_logits, gt_masks)

            dice_loss, iou_loss = dice_iou_loss(coarse_pre, gt_masks)

            loss = bce_loss + 0.5 * (0.5*dice_loss + 0.5*iou_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch:03d} | Avg loss: {avg_loss:.6f}")

        # 4. 儲存最佳模型 --------------------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pth"))
            print(f"  ✅  Saved new best model (loss={best_loss:.6f})")
            print(f"Epoch {epoch} | BCE: {bce_loss.item():.4f} | Dice: {dice_loss.item():.4f} | IoU: {iou_loss.item():.4f} | Total: {loss.item():.4f}")


        # 5. 可視化 --------------------------------------------------------
        model.eval()
        with torch.no_grad():
            sample = next(iter(dataloader))
            imgs = sample["image"].to(DEVICE)

            probs = model(imgs, use_refiner=False)["coarse_mask"][:4]
            probs  = probs.cpu()

            # 列印 min/max
            #print(f"  pred min/max: {probs.min():.4f}  {probs.max():.4f}")

            # 存圖（整個 batch 做一張 grid；nrow 可依 batch_size 調整）
            save_image(
                probs,
                os.path.join(PRED_DIR, f"epoch_{epoch:03d}.png"),
                nrow=4,
                padding=1
            )
            
        # ==== 寫入 log 檔案 ====
        with open(log_path, "a") as f:
            f.write(f"{epoch},{loss.item():.6f},{bce_loss.item():.6f},{dice_loss.item():.6f},{iou_loss.item():.6f},{probs.min():.4f},{probs.max():.4f}\n")

if __name__ == "__main__":
    print("✅ 使用設備：", DEVICE)
    print("CUDA 是否可用：", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 名稱：", torch.cuda.get_device_name(0))

    train()
 
# python pretrain.py