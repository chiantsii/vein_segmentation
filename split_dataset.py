import os
import shutil
import random

# 原始資料位置
SOURCE_DIR = "data/LVD2021/36_Holly_labels/all"
OUTPUT_ROOT = "data/LVD2021/36_Holly_labels"

# 目標子資料夾
PRETRAIN_DIR = os.path.join(OUTPUT_ROOT, "pretrain")
TRAIN_DIR = os.path.join(OUTPUT_ROOT, "train")
VALID_DIR = os.path.join(OUTPUT_ROOT, "valid")
TEST_DIR  = os.path.join(OUTPUT_ROOT, "test")

PRETRAIN_NUM = 40
VALID_RATIO = 0.2
TEST_RATIO = 0.2

def split_dataset():
    # 建立目標資料夾
    for path in [PRETRAIN_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR]:
        os.makedirs(path, exist_ok=True)

    # 所有樣本（子資料夾名稱）
    all_samples = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])
    print(f"總樣本數: {len(all_samples)}")

    # 隨機打散順序
    random.shuffle(all_samples)

    # 分割
    pretrain_samples = all_samples[:PRETRAIN_NUM]
    remaining = all_samples[PRETRAIN_NUM:]

    num_valid = int(len(remaining) * VALID_RATIO)
    num_test = int(len(remaining) * TEST_RATIO)

    valid_samples = remaining[:num_valid]
    test_samples = remaining[num_valid:num_valid + num_test]
    train_samples = remaining[num_valid + num_test:]

    # 搬移 function
    def copy_samples(samples, target_dir):
        for sample in samples:
            src = os.path.join(SOURCE_DIR, sample)
            dst = os.path.join(target_dir, sample)
            shutil.copytree(src, dst)

    copy_samples(pretrain_samples, PRETRAIN_DIR)
    copy_samples(train_samples, TRAIN_DIR)
    copy_samples(valid_samples, VALID_DIR)
    copy_samples(test_samples, TEST_DIR)

    # log
    print(f"完成分割：")
    print(f"  Pretrain: {len(pretrain_samples)}")
    print(f"  Train:    {len(train_samples)}")
    print(f"  Valid:    {len(valid_samples)}")
    print(f"  Test:     {len(test_samples)}")

if __name__ == "__main__":
    split_dataset()


# python split_dataset.py