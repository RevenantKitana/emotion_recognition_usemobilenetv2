import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Tạo thư mục lưu kết quả nếu chưa tồn tại
os.makedirs("results", exist_ok=True)

def plot_training_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    best_acc = max(val_acc)
    best_epoch = val_acc.index(best_acc) + 1
    best_loss = min(val_loss)

    print(f"✅ Best Validation Accuracy: {best_acc:.4f} at Epoch {best_epoch}")
    print(f"✅ Lowest Validation Loss: {best_loss:.4f}")

    # Biểu đồ Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(acc, label='Train Accuracy', marker='o')
    plt.plot(val_acc, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/accuracy_plot.png', dpi=300)
    plt.close()

    # Biểu đồ Loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label='Train Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/loss_plot.png', dpi=300)
    plt.close()


def evaluate_model(model, validation_dataset):
    y_true = []
    y_pred = []

    # Dự đoán từng batch
    for images, labels in validation_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_labels = getattr(validation_dataset, 'class_names', [str(i) for i in range(np.max(y_true) + 1)])

    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print("📊 Classification Report:")
    print(report)

    # Lưu classification report
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()

    # Phân tích lớp bị nhầm nhiều nhất
    if cm.shape[0] > 1:
        error_matrix = cm.copy()
        np.fill_diagonal(error_matrix, 0)
        max_confused_class = np.argmax(np.sum(error_matrix, axis=1))
        confused_with = np.argmax(error_matrix[max_confused_class])
        print(f"⚠️ Lớp '{class_labels[max_confused_class]}' thường bị nhầm với '{class_labels[confused_with]}' "
              f"({error_matrix[max_confused_class][confused_with]} lần)")
