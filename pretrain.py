from core_net import CoRE_Net
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from dataloader import VeinSegmentationDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()
    decoder = Decoder()
    point_refiner = PointRefiner()
    model = CoRE_Net(encoder, decoder, point_refiner).to(device)

    # 鎖定 Point Refiner 的參數
    for param in model.point_refiner.parameters():
        param.requires_grad = False

    # 只訓練 encoder + decoder
    params_to_optimize = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=1e-4)
    criterion = nn.BCELoss()

    dataset = VeinSegmentationDataset("/Users/chiantsii/Desktop/vein_seg/data/LVD2021/36_Holly_labels/pretrain")
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    best_loss = float('inf')  # 初始為無限大

    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            images = batch["image"].to(device)
            gt_masks = batch["gt_mask"].to(device)
            leaf_masks = batch["leaf_mask"].to(device)

            # ✅ 使用 coarse mask，關閉 point refiner
            output = model(images, use_refiner=False)
            coarse_mask = output["coarse_mask"]

            loss_map = criterion(coarse_mask, gt_masks)
            masked_loss = (loss_map * leaf_masks).mean() # 只有葉片範圍

            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()

            epoch_loss += masked_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"✅ Saved new best model at epoch {epoch} with loss {best_loss:.4f}")
        
        model.eval()
        with torch.no_grad():
            sample = next(iter(dataloader))
            sample_imgs = sample["image"].to(device)
            output = model(sample_imgs, use_refiner=False)
            preds = output["coarse_mask"] > 0.5
            save_image(preds, f"predictions/epoch_{epoch:02d}.png")
            

if __name__ == "__main__":
    train()

# python pretrian.py