
import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.utils import save_image
from core_net import CoRE_Net
from encoder import Encoder
from decoder import Decoder
from point_refiner.point_refiner import PointRefiner
from torchvision import transforms

def load_model(checkpoint_path, device):
    model = CoRE_Net(Encoder(), Decoder(), PointRefiner())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_pseudo_labels(model, unlabeled_dir, output_dir, epoch, device):
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])

    for root, _, files in os.walk(unlabeled_dir):
        for file in tqdm(files):
            if "背景" not in file:
                continue

            img_path = os.path.join(root, file)
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

            with torch.no_grad():
                output = model(input_tensor)
                pred = output["coarse_mask"]  # [1, 1, H, W]

            prefix = file.replace("_背景.jpg", "")
            save_subdir = os.path.join(output_dir, prefix)
            os.makedirs(save_subdir, exist_ok=True)

            save_path = os.path.join(save_subdir, f"{prefix}_pseudo_epoch{epoch}.png")
            save_image(pred, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--unlabeled_dir", type=str, required=True, help="Path to unlabeled background images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save pseudo labels")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number for naming")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    generate_pseudo_labels(model, args.unlabeled_dir, args.output_dir, args.epoch, device)
