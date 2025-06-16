import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return 1 - (intersection + smooth) / (union + smooth)


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

    pred_mask = output["coarse_mask"]
    bce = F.binary_cross_entropy(pred_mask, output["pseudo_label"])
    dice = dice_loss(pred_mask, output["pseudo_label"])
    iou  = iou_loss(pred_mask, output["pseudo_label"])

    pixel_loss = bce + 0.5 * dice + 0.5 * iou  # 加總 loss

    # Point-wise BCE Loss
    point_loss = 0
    for i in range(len(output["point_logits"])):
        logits = output["point_logits"][i]             # (K_i, 1)
        coords = output["point_coords"][i].long()      # (K_i, 2)

        gt_i = gt_mask[i, 0]                           # [H, W]

        # 防止超出邊界
        coords[:, 0] = coords[:, 0].clamp(0, gt_i.shape[0] - 1)
        coords[:, 1] = coords[:, 1].clamp(0, gt_i.shape[1] - 1)

        labels = gt_i[coords[:, 0], coords[:, 1]]      # (K_i,)
        logits = logits.squeeze()                      # (K_i,)
        point_loss += F.binary_cross_entropy_with_logits(logits, labels)

    total_loss = pixel_loss + point_loss
    return total_loss, {"pixel_loss": pixel_loss.item(), "point_loss": point_loss.item()}


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