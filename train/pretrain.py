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

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    dataset = VeinSegmentationDataset("/Users/chiantsii/Desktop/vein_seg/data/LVD2021/36_Holly_labels/train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    best_loss = float('inf')  # 初始為無限大

    for epoch in range(1, 31):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            images = batch["image"].to(device)
            gt_masks = batch["gt_mask"].to(device)
            leaf_masks = batch["leaf_mask"].to(device)

            output = model(images)
            coarse_mask = output["coarse_mask"]

            loss_map = criterion(coarse_mask, gt_masks)
            masked_loss = (loss_map * leaf_masks).mean()

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
        

        # 儲存預測圖
        model.eval()
        with torch.no_grad():
            sample = next(iter(dataloader))
            sample_imgs = sample["image"].to(device)
            output = model(sample_imgs)
            preds = output["coarse_mask"]
            save_image(preds, f"predictions/epoch_{epoch:02d}.png")

        torch.save(model.state_dict(), f"checkpoints/core_net_epoch{epoch:02d}.pth")

if __name__ == "__main__":
    train()
