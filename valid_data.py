import os
import random
import shutil

def split_test_to_valid(test_dir, valid_dir, num_samples=20, seed=42):
    """
    從指定的 test_dir 根目錄中隨機取 num_samples 個子資料夾，
    並將整個子資料夾移動到 valid_dir 下，保留結構。
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    os.makedirs(valid_dir, exist_ok=True)

    # 取得所有直接子資料夾
    subdirs = [d for d in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, d))]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {test_dir}")

    random.seed(seed)
    selected = random.sample(subdirs, min(num_samples, len(subdirs)))

    for sub in selected:
        src_path = os.path.join(test_dir, sub)
        dst_path = os.path.join(valid_dir, sub)
        # 移動子資料夾
        shutil.move(src_path, dst_path)
        print(f"Moved folder {sub} to {valid_dir}.")

    print(f"Moved {len(selected)} folders from {test_dir} to {valid_dir}.")

# ======= 在此直接設定參數 =======
TEST_DIR = "data/LVD2021/36_Holly_labels/test"
VALID_DIR = "data/LVD2021/36_Holly_labels/valid"
NUM_SAMPLES = 20
SEED = 42
# ===============================

if __name__ == '__main__':
    split_test_to_valid(TEST_DIR, VALID_DIR, NUM_SAMPLES, SEED)

# python valid_data.py