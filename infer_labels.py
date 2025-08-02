import os
import torch
import numpy as np
import cv2
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from core_net import CoRE_Net
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from dataloader import VeinSegmentationDataset


def infer_labels(model, dataset, device, save_dir=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    pseudo_labels = []
    b_maps = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = batch["image"].to(device)
            output = model(img, use_refiner=True)

            # --- 1. pseudo label: 來自 coarse 預測 ---
            coarse_prob = torch.sigmoid(output["coarse_logits"]).squeeze().cpu().numpy()  # shape (H, W)

            # normalize → OTSU
            norm = (coarse_prob - coarse_prob.min()) / (coarse_prob.max() - coarse_prob.min() + 1e-8)
            norm_uint8 = (norm * 255).astype(np.uint8)
            _, pseudo = cv2.threshold(norm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pseudo = torch.tensor(pseudo / 255.0).unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]

            # --- 2. b_map：高或低 confidence 區域 ---
            b_map = torch.zeros_like(pseudo)
            coarse_prob_tensor = torch.tensor(coarse_prob).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]
            b_map = torch.zeros_like(coarse_prob_tensor)
            b_map[(coarse_prob_tensor >= 0.9) | (coarse_prob_tensor <= 0.1)] = 1.0

            # --- 3. 儲存與回傳 ---
            pseudo_labels.append(pseudo)
            b_maps.append(b_map)

            if save_dir:
                bg_path = dataset.samples[i][0]
                sample_id = os.path.basename(os.path.dirname(bg_path))
                sample_dir = os.path.join(save_dir, sample_id)
                os.makedirs(sample_dir, exist_ok=True)

                save_image(pseudo[0], os.path.join(sample_dir, "pseudo.png"))
                save_image(b_map[0], os.path.join(sample_dir, "b_map.png"))
                save_image(img[0].cpu(), os.path.join(sample_dir, "image.png"))

                if "0003" in bg_path:
                    fig9_dir = f"debug_vis/fig9_samples"
                    os.makedirs(fig9_dir, exist_ok=True)
                    save_image(pseudo[0], os.path.join(fig9_dir, f"epoch{save_dir.split('_')[-1]}_0003_pseudo.png"))
                    save_image(img[0].cpu(), os.path.join(fig9_dir, f"input_0003.png"))

    dataset.masks = pseudo_labels
    dataset.b_maps = b_maps
    return dataset


if __name__ == "__main__":
    PRETRAIN_CKPT = "checkpoints/36_Holly_labels/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CoRE_Net(Encoder(), Decoder(), PointRefiner()).to(device)
    model.load_state_dict(torch.load(PRETRAIN_CKPT, map_location=device))
    model.eval()

    dataset = VeinSegmentationDataset(
        "data/LVD2021/36_Holly_labels/train",
        image_size=(448, 448)
    )

    pseudo_dir = "data/LVD2021/36_Holly_labels/pseudo/epoch_1_coarse"
    infer_labels(model, dataset, device, save_dir=pseudo_dir)


# python infer_labels.py