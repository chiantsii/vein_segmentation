import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import VeinSegmentationDataset
from core_net import CoRE_Net
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from core_net_loss import core_net_loss


def train_self_training(data_root, num_epochs=100, batch_size=5, lr=1e-4, checkpoint_path="checkpoints/self_train.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()
    decoder = Decoder()
    point_refiner = PointRefiner()
    model = CoRE_Net(encoder, decoder, point_refiner).to(device)

    pretrain_ckpt = "checkpoints/best_model.pth"
    if os.path.exists(pretrain_ckpt):
        model.load_state_dict(torch.load(pretrain_ckpt))
        print(f"📥 Loaded pretrained weights from {pretrain_ckpt}")
    else:
        print("⚠️ No pretrained weights found. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = VeinSegmentationDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        saved_count = 0  # 控制每個 epoch 只存前兩張
        save_dir = f"predictions/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            images = batch["image"].to(device)

            # forward
            output = model(images)

            # generate pseudo label from coarse mask (即時產生)
            with torch.no_grad():
                pseudo_gt = (output["coarse_mask"]).detach()

                # === 儲存前兩張預測結果 ===
                for i in range(images.size(0)):
                    if saved_count >= 2:
                        break

                    # 儲存 coarse mask
                    coarse_mask = output["coarse_mask"][i]
                    torchvision.utils.save_image(coarse_mask, os.path.join(save_dir, f"sample_{saved_count}_coarse.png"))

                    # 儲存 refined mask（如果有）
                    if "refined_mask" in output:
                        refined_mask = output["refined_mask"][i]
                        torchvision.utils.save_image(refined_mask, os.path.join(save_dir, f"sample_{saved_count}_refined.png"))

                    # 儲存 point mask（如果有）
                    if "point_mask" in output:
                        point_mask = output["point_mask"][i]
                        torchvision.utils.save_image(point_mask, os.path.join(save_dir, f"sample_{saved_count}_pointmask.png"))

                    saved_count += 1

            # loss with pseudo supervision
            loss, loss_dict = core_net_loss(output, pseudo_gt)

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
    train_self_training(data_root="data/LVD2021/36_Holly_labels/train")


# python self_train_pure.py