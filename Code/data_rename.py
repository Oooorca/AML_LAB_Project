import os
import random

# 数据集路径
base_dir = r"D:\forStudy\AMLlab\Participant2"

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

for grasp in grasp_folders:
    grasp_path = os.path.join(base_dir, grasp)
    if not os.path.isdir(grasp_path):
        continue

    for subfolder in os.listdir(grasp_path):
        subfolder_path = os.path.join(grasp_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # 获取所有 CSV 文件
        csv_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.csv')]
        
        if len(csv_files) != 15:
            print(f"Warning: {subfolder_path} does not have exactly 15 CSV files.")
            continue

        # 打乱顺序
        random.shuffle(csv_files)

        # 重命名为 1.csv ~ 15.csv
        for i, old_name in enumerate(csv_files, start=1):
            old_path = os.path.join(subfolder_path, old_name)
            new_path = os.path.join(subfolder_path, f"{i}.csv")
            os.rename(old_path, new_path)

        print(f"Renamed CSVs in: {subfolder_path}")
