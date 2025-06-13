
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

def resize_with_padding(img, target_size=(448, 448)):
    old_size = img.size  # (width, height)
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.BILINEAR)

    new_img = Image.new("RGB", target_size)
    paste_position = ((target_size[0]-new_size[0])//2, (target_size[1]-new_size[1])//2)
    new_img.paste(img, paste_position)
    return new_img

def generate_pseudo_labels(model, unlabeled_dir, output_dir, epoch, device):
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])

    os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

    for root, _, files in os.walk(unlabeled_dir):
        for file in tqdm(files):
            if "背景" not in file:
                continue

            img_path = os.path.join(root, file)
            image = Image.open(img_path).convert("RGB")
            image = resize_with_padding(image, target_size=(448, 448))
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred = output["coarse_mask"]

            # 提取 leaf 編號作為檔名：如 133_0003
            leaf_id = os.path.splitext(file)[0].replace("_背景", "")
            save_path = os.path.join(output_dir, f"{leaf_id}_pseudo_epoch{epoch}.png")

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


# python generate_pseudo_labels.py \
#   --model checkpoints/best_model.pth \
#   --unlabeled_dir data/LVD2021/36_Holly_labels/train \
#   --output_dir data/LVD2021/36_Holly_labels/train/pseudo_labels \
#   --epoch 30