import pandas as pd
import matplotlib.pyplot as plt

csv_path = "checkpoints/36_Holly_labels/loss_log.csv"
df = pd.read_csv(csv_path)

# 只取每個 outer epoch 的最後一輪 inner_epoch==9
df_last = df[df["inner_epoch"] == 9].copy()
df_last["epoch"] = df_last["outer_epoch"]

# 計算指標（1 - loss）
dice = 1 - df_last["dice_loss"]
iou  = 1 - df_last["iou_loss"]

plt.figure(figsize=(10, 5))
plt.plot(df_last["epoch"], dice, label="Dice", marker="o")
plt.plot(df_last["epoch"], iou, label="IoU", marker="s")

plt.title("Segmentation Metrics per Outer Epoch")
plt.xlabel("Outer Epoch")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("checkpoints/36_Holly_labels/outer_epoch_metrics.png")
plt.show()


# python plot_metrics.py

