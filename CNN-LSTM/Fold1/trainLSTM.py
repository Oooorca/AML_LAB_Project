import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

OUTPUT_DIR = "./LSTM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 10
STEP_SIZE = 5
NUM_CLASSES = 8
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3

LABEL_PATH = "grasp_labels_stable.csv"

def sliding_window_subsequences(data, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    subseqs = []
    T = len(data)
    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        subseqs.append(data[start:end])
    return subseqs

def load_dataset(train_dir="Participant1", test_dir="Participant2"):
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

    def load_from_dir(base_dir):
        X_list, y_list = [], []
        for label_name in grasp_folders:
            folder_path = os.path.join(base_dir, label_name)
            if not os.path.isdir(folder_path):
                continue

            subfolders = [sf for sf in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, sf))]

            for subf in subfolders:
                subf_path = os.path.join(folder_path, subf)
                csv_files = glob.glob(os.path.join(subf_path, '*.csv'))

                for file_path in csv_files:
                    df = pd.read_csv(file_path, header=None)
                    df.columns = ['time','flex1','flex2','flex3','flex4','flex5','flex6']
                    data = df[['flex1','flex2','flex3','flex4','flex5','flex6']].values
                    subseqs = sliding_window_subsequences(data, WINDOW_SIZE, STEP_SIZE)
                    for subseq in subseqs:
                        X_list.append(subseq)
                        y_list.append(label_name)

        return np.array(X_list), np.array(y_list)

    X_trainval, y_trainval = load_from_dir(train_dir)
    X_test, y_test = load_from_dir(test_dir)

    le = LabelEncoder()
    y_trainval_enc = le.fit_transform(y_trainval)
    y_trainval_onehot = to_categorical(y_trainval_enc, num_classes=NUM_CLASSES)

    y_test_enc = le.transform(y_test)
    y_test_onehot = to_categorical(y_test_enc, num_classes=NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval_onehot, test_size=0.1111, random_state=42,
        stratify=y_trainval_enc
    )

    print("Train/Val/Test dataset shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test_onehot.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test_onehot, le

def build_lstm_model(input_shape=(WINDOW_SIZE, 6), num_classes=NUM_CLASSES):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def plot_confusion_matrix(y_true, y_pred, label_path=LABEL_PATH, normalize=True, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    labels = pd.read_csv(label_path)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=grasp_names, yticklabels=grasp_names)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.933, top=0.933)
    plt.savefig(save_path, dpi=300)
    print(f"Confusion Matrix saved to {save_path}")
    plt.show()

def generate_classification_report(y_true, y_pred, label_path=LABEL_PATH,
                                   save_csv_path="classification_report.csv",
                                   save_img_path="classification_report.png"):
    labels = pd.read_csv(label_path)
    labels = labels.sort_values('Label')  
    grasp_names = labels['Grasp Type'].tolist()

    report_dict = classification_report(y_true, y_pred, target_names=grasp_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).T
    df_report.to_csv(save_csv_path, index=True)
    print(f"Classification report saved to {save_csv_path}")

    plt.figure(figsize=(10, 6))
    df_report = df_report.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df_report = df_report.drop(columns=['support'], errors='ignore')

    sns.heatmap(df_report.astype(float), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.9)
    plt.savefig(save_img_path, dpi=300)
    print(f"Classification report image saved to {save_img_path}")
    plt.show()

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_dataset()
    model = build_lstm_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    generate_classification_report(y_true, y_pred,
                                   save_csv_path=os.path.join(OUTPUT_DIR, "classification_report.csv"),
                                   save_img_path=os.path.join(OUTPUT_DIR, "classification_report.png"))

if __name__ == "__main__":
    main()
