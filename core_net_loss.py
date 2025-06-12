import torch
import torch.nn.functional as F

def core_net_loss(output, gt_mask):
    """
    Args:
        output: dict from CoRE-Net forward()
            - "coarse_mask": [B, 1, H, W]
            - "point_logits": list of [K_i, 1]
            - "point_coords": list of [K_i, 2]
        gt_mask: ground truth mask, shape [B, 1, H, W]

    Returns:
        total_loss: scalar
        loss_dict: dict with 'pixel_loss', 'point_loss'
    """

    # --- Loss 1: pixel-wise BCE ---
    pixel_loss = F.binary_cross_entropy(output["coarse_mask"], gt_mask)

    # --- Loss 2: point-wise BCE (每張圖的點)
    point_loss = 0
    for i in range(len(output["point_logits"])):
        logits = output["point_logits"][i]             # (K_i, 1)
        coords = output["point_coords"][i].long()      # (K_i, 2), type long for indexing

        # 抽出該圖的對應 GT mask
        gt_i = gt_mask[i, 0]                           # [H, W]
        H, W = gt_i.shape

        # 取得每個點的 ground truth label
        clamped_coords = torch.clamp(coords, 0, W - 1)
        labels = gt_i[clamped_coords[:, 0], clamped_coords[:, 1]]  # shape: (K_i,)
        labels = labels.unsqueeze(1)                               # (K_i, 1)

        # 計算 BCE loss
        point_loss += F.binary_cross_entropy_with_logits(logits, labels)

    point_loss = point_loss / len(output["point_logits"])  # 平均每張圖的點 loss

    # 總和 loss，可自行調整權重
    total_loss = pixel_loss + point_loss

    return total_loss, {
        "pixel_loss": pixel_loss.item(),
        "point_loss": point_loss.item()
    }

if __name__ == "__main__":
    # ==== 模擬輸入 ====
    B, H, W = 2, 448, 448  # Batch size 2, image size 64x64

    # 模擬 coarse mask output 和 ground truth
    coarse_mask = torch.rand(B, 1, H, W)
    gt_mask = torch.randint(0, 2, (B, 1, H, W)).float()

    # 模擬每張圖挑的點 (每張圖各挑 5 個點)
    point_logits = [torch.randn(5, 1) for _ in range(B)]
    point_coords = [torch.randint(0, H, (5, 2)) for _ in range(B)]  # 隨機產生點座標

    # 包裝 output 結構
    output = {
        "coarse_mask": coarse_mask,
        "point_logits": point_logits,
        "point_coords": point_coords,
    }

    # ==== 呼叫 loss 函數 ====
    loss, loss_dict = core_net_loss(output, gt_mask)

    # ==== 顯示結果 ====
    print("Total Loss:", loss.item())
    print("Pixel Loss:", loss_dict["pixel_loss"])
    print("Point Loss:", loss_dict["point_loss"])