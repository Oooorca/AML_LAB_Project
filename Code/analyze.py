import matplotlib.pyplot as plt
import numpy as np

# 模型 & 标签
models = ["CNN", "LSTM", "CNN-BiLSTM", "SVM", "NN"]
grasp_types = [
    "Adduction", "Fixed Hook", "Large\nDiameter", "Parallel\nExtension",
    "Precision\nSphere", "Sphere\n4-Finger", "Ventral", "Writing\nTripod"
]

# Accuracy values per label per fold (shape: [8 grasp types][5 models])
fold1_acc = np.array([
    [0.73, 0.71, 0.80, 0.99, 0.99],
    [0.55, 0.55, 0.73, 0.76, 0.88],
    [0.75, 0.69, 0.78, 0.91, 0.89],
    [0.80, 0.78, 0.84, 0.71, 0.67],
    [0.32, 0.46, 0.60, 0.64, 0.40],
    [0.53, 0.47, 0.54, 0.86, 0.94],
    [0.70, 0.77, 0.82, 0.99, 0.96],
    [0.64, 0.78, 0.82, 0.99, 0.70]
])
fold2_acc = np.array([
    [0.53, 0.52, 0.60, 0.90, 0.79],
    [0.44, 0.48, 0.55, 0.72, 0.56],
    [0.78, 0.66, 0.76, 0.98, 0.92],
    [0.68, 0.67, 0.73, 0.33, 0.67],
    [0.57, 0.53, 0.68, 0.68, 0.87],
    [0.52, 0.43, 0.61, 0.71, 0.70],
    [0.62, 0.66, 0.70, 1.00, 0.98],
    [0.71, 0.76, 0.86, 0.63, 0.48]
])

# Mean and error (difference between folds) per model per grasp type
mean_acc = (fold1_acc + fold2_acc) / 2
std_acc = np.abs(fold1_acc - fold2_acc)

x = np.arange(len(grasp_types))
bar_width = 0.15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.figure(figsize=(13, 6))
for i in range(len(models)):
    plt.bar(x + i * bar_width - 0.3, mean_acc[:, i], yerr=std_acc[:, i],
            width=bar_width, label=models[i], color=colors[i], capsize=4)

plt.xticks(x, grasp_types, rotation=30)
plt.ylabel("Mean Accuracy with Error Bars (|Fold1 - Fold2|)")
plt.title("Model Generalization in LOSO-CV: Per-Class Accuracy with Error Bars")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()