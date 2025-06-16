
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import VeinSegmentationDataset
from core_net import CoRE_Net
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from loss import core_net_loss

def train_self_training(data_root, num_epochs=100, batch_size=5, lr=1e-4, checkpoint_path="checkpoints/self_train.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()
    decoder = Decoder()
    point_refiner = PointRefiner()
    model = CoRE_Net(encoder, decoder, point_refiner).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = VeinSegmentationDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            images = batch["image"].to(device)
            gt_masks = batch["gt_mask"].to(device)

            output = model(images)
            loss, loss_dict = core_net_loss(output, gt_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Pixel: {loss_dict['pixel_loss']:.4f} | Point: {loss_dict['point_loss']:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved best model at epoch {epoch} with loss {best_loss:.4f}")

if __name__ == "__main__":
    train_self_training(data_root="data/LVD2021/36_Holly_labels/self_train")
