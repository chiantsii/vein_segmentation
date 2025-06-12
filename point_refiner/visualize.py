import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_point_coords(prob_map, point_coords, title='Points on Image'):
    """
    可視化 Top-K + PCM 選出點的位置
    Args:
        prob_map: Tensor (1, 1, H, W) - e.g. decoder 的 sigmoid 輸出
        point_coords: Tensor (K, 2) - 每個點的 (i, j)
        title: 標題
    """
    prob_map_np = prob_map.squeeze().detach().cpu().numpy()  # shape: (H, W)
    coords_np = point_coords.detach().cpu().numpy()          # shape: (K, 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(prob_map_np, cmap='gray')
    plt.scatter(coords_np[:, 1], coords_np[:, 0], s=10, c='red', marker='x')  # 注意: j=x, i=y
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from point_refiner import PointRefiner

    # 假輸入
    prob = torch.sigmoid(torch.randn(1, 1, 448, 448))
    mask = (prob > 0.5).float()
    feat = torch.randn(1, 256, 14, 14)

    pr = PointRefiner()
    logits, coords = pr(prob, mask, feat)

    visualize_point_coords(prob, coords[0], title='Selected Points by CMM + PCM')
