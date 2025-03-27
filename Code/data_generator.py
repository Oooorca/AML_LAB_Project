import os
import random
import numpy as np
import pandas as pd

# 原数据路径
base_dir = r"D:\forStudy\AMLlab\Participant1"
# 目标路径
target_dir = r"D:\forStudy\AMLlab\Participant2"

# 8种grasp类型
grasp_folders = [
    "Adduction_Grip",
    "Fixed_hook",
    "Large_diameter",
    "Parallel_extension",
    "Precision_sphere",
    "Sphere_4_finger",
    "Ventral",
    "Writing_Tripod"
]

# 创建目标根目录
os.makedirs(target_dir, exist_ok=True)

for grasp in grasp_folders:
    grasp_path = os.path.join(base_dir, grasp)
    if not os.path.isdir(grasp_path):
        print(f"Skipping {grasp_path}, not a directory.")
        continue
    
    target_grasp_path = os.path.join(target_dir, grasp)
    os.makedirs(target_grasp_path, exist_ok=True)

    for subfolder in os.listdir(grasp_path):
        subfolder_path = os.path.join(grasp_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        target_subfolder_path = os.path.join(target_grasp_path, subfolder)
        os.makedirs(target_subfolder_path, exist_ok=True)

        for filename in os.listdir(subfolder_path):
            if not filename.lower().endswith(".csv"):
                continue

            old_path = os.path.join(subfolder_path, filename)

            try:
                # 读取CSV
                data = pd.read_csv(old_path, header=None)
                n_rows = len(data)

                # 如果行数不够，跳过
                if n_rows <= 10:
                    print(f"Skipping {old_path}, not enough rows.")
                    continue

                # 随机上移行数
                shift_rows = random.randint(1, 10)

                # 上移第2~7列数据
                for c in range(1, 7):
                    data.iloc[0:n_rows - shift_rows, c] = data.iloc[shift_rows:, c].values

                # 删除末尾行（整行删除）
                data = data.iloc[0:n_rows - shift_rows]

                # 加减随机整数（只针对第2~7列）
                delta = random.randint(-10, 0)
                data.iloc[:, 1:] = data.iloc[:, 1:] + delta

                # 强制所有值为整数（四舍五入后转int）
                data = data.round().astype(int)

                # 保存
                new_filename = f"{os.path.splitext(filename)[0]}_aug.csv"
                new_path = os.path.join(target_subfolder_path, new_filename)
                data.to_csv(new_path, index=False, header=False)

                print(f"Augmented and saved: {old_path} -> {new_path}")

            except Exception as e:
                print(f"Error processing {old_path}: {e}")
