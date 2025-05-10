import sys
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Thêm sys.path cho utils nếu cần
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from preprocess import preprocess_image

# Danh sách nhãn cảm xúc
emotion_labels = ['Ang', 'Dis', 'Fea', 'Hap', 'Neu', 'Sad', 'Sur']

# Biến toàn cục chứa model hiện tại
current_model = None

# Hàm load model mặc định
def load_default_model():
    global current_model
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_model.keras'))
    try:
        current_model = load_model(default_path)
        model_label.config(text=f"Model: {os.path.basename(default_path)}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load default model:\n{str(e)}")
        sys.exit(1)

# Hàm cho phép người dùng chọn mô hình khác
def select_model_and_load():
    global current_model
    file_path = filedialog.askopenfilename(
        title="Select Model File (.h5 or .keras)",
        filetypes=[("Model", "*.keras *.h5")]
    )
    if not file_path:
        return
    try:
        model = load_model(file_path)
        current_model = model
        model_label.config(text=f"Model: {os.path.basename(file_path)}")
        messagebox.showinfo("Model Loaded", "Model loaded successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

# Hàm dự đoán cảm xúc
def predict_emotion():
    button.config(state="disabled")
    stats_label.config(text="Processing... Please wait.")

    def process_images():
        image_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not image_paths:
            stats_label.config(text="Emotion Distribution: No images selected")
            button.config(state="normal")
            return

        for row in tree.get_children():
            tree.delete(row)

        emotion_counts = {label: 0 for label in emotion_labels}
        total_images = 0
        all_probabilities = []

        for image_path in image_paths:
            try:
                image_name = os.path.basename(image_path)
                img = preprocess_image(image_path, img_size=224)
                prediction = current_model.predict(img, verbose=0)[0]
                emotion_index = np.argmax(prediction)
                emotion = emotion_labels[emotion_index]
                probabilities = prediction * 100
                prob_str = ", ".join([f"{label}: {prob:.2f}%" for label, prob in zip(emotion_labels, probabilities)])
                all_probabilities.append(probabilities)
                window.after(0, lambda name=image_name, emo=emotion, prob=prob_str:
                             tree.insert("", "end", values=(name, emo, prob)))
                emotion_counts[emotion] += 1
                total_images += 1
            except Exception as e:
                window.after(0, lambda name=image_name, err=str(e):
                             tree.insert("", "end", values=(name, f"Error: {err}", "")))

        if total_images > 0:
            emotion_percentages = {label: (count / total_images) * 100 for label, count in emotion_counts.items()}
            dist_str = "Emotion Distribution: " + ", ".join(
                [f"{label}: {percent:.2f}%" for label, percent in emotion_percentages.items()])
            avg_probabilities = np.mean(all_probabilities, axis=0)
            avg_str = "Average Probabilities: " + ", ".join(
                [f"{label}: {prob:.2f}%" for label, prob in zip(emotion_labels, avg_probabilities)])
            stats_str = f"{dist_str}\n{avg_str}"
            window.after(0, lambda: stats_label.config(text=stats_str))
        else:
            window.after(0, lambda: stats_label.config(text="Emotion Distribution: No valid images processed"))
        window.after(0, lambda: button.config(state="normal"))

    thread = threading.Thread(target=process_images)
    thread.start()

# Giao diện GUI
window = tk.Tk()
window.title("Emotion Recognition")
window.geometry("800x600")

frame = tk.Frame(window)
frame.pack(pady=10)

# Hiển thị tên mô hình hiện tại
model_label = tk.Label(frame, text="Model: Loading...", fg="blue")
model_label.pack(pady=5)

# Nút chọn model
select_model_btn = tk.Button(frame, text="Select Model", command=select_model_and_load)
select_model_btn.pack(pady=5)

# Nút upload ảnh
button = tk.Button(frame, text="Upload Images", command=predict_emotion)
button.pack(pady=5)

# Label thống kê
stats_label = tk.Label(frame, text="Emotion Distribution:", wraplength=750, justify="left")
stats_label.pack(pady=5)

# Treeview kết quả
columns = ("Image Name", "Predicted Emotion", "Probabilities (%)")
tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)
for col in columns:
    tree.heading(col, text=col)
tree.column("Image Name", width=150)
tree.column("Predicted Emotion", width=150)
tree.column("Probabilities (%)", width=500)
tree.pack(pady=5)

scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Load model mặc định khi khởi động
load_default_model()

# Start GUI loop
window.mainloop()
