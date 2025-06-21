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


# -------------------------- 參數 --------------------------
which       = '36_Holly_labels'
DATA_ROOT   = "/Users/chiantsii/Desktop/vein_seg/data/LVD2021/" + which + "/pretrain"
BATCH_SIZE  = 4
EPOCHS      = 100
LR          = 0.001
CKPT_DIR    = "checkpoints/" + which
PRED_DIR    = "predictions/" + which
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_ROOT    = "/Users/chiantsii/Desktop/vein_seg/data/LVD2021/" + which + "/pretrain"
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
    pos_weight = torch.tensor([5.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 2. 資料 -------------------------------------------------------------
    dataset   = VeinSegmentationDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    best_loss = float("inf")
    
    # csv 記錄 avg loss / max / min
    log_path = os.path.join(CKPT_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,avg_loss,pred_min,pred_max\n")

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

            # per-pixel BCE，再乘 leaf_masks 僅計算葉片區域
            loss_map     = criterion(coarse_logits, gt_masks)
            masked_loss  = (loss_map * leaf_masks).mean()

            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()

            epoch_loss += masked_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch:03d} | Avg loss: {avg_loss:.6f}")

        # 4. 儲存最佳模型 --------------------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pth"))
            print(f"  ✅  Saved new best model (loss={best_loss:.6f})")

        # 5. 可視化 --------------------------------------------------------
        model.eval()
        with torch.no_grad():
            sample = next(iter(dataloader))
            imgs = sample["image"].to(DEVICE)

            probs = model(imgs, use_refiner=False)["coarse_mask"]
            probs  = probs.cpu()

            # 列印 min/max
            print(f"  pred min/max: {probs.min():.4f}  {probs.max():.4f}")

            # 存圖（整個 batch 做一張 grid；nrow 可依 batch_size 調整）
            save_image(
                probs,
                os.path.join(PRED_DIR, f"epoch_{epoch:03d}.png"),
                nrow=4,
                padding=2
            )
            
        # ==== 寫入 log 檔案 ====
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{probs.min():.4f},{probs.max():.4f}\n")

if __name__ == "__main__":
    train()
 
# python pretrain.py