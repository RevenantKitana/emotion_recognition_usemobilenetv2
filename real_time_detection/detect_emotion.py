import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

def choose_model():
    """Hiển thị hộp thoại chọn mô hình"""
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Chọn file mô hình (.keras hoặc .h5)",
        filetypes=[("Keras model files", "*.keras *.h5")]
    )
    return path

# Danh sách các nhãn cảm xúc
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Load bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Chọn mô hình lần đầu
model_path = choose_model()
if not model_path:
    print("Không có mô hình được chọn.")
    exit()

model = load_model(model_path)
print(f"Đã load mô hình: {model_path}")

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = face / 255.0

        # Dự đoán cảm xúc
        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Vẽ khung và hiển thị cảm xúc
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị lên cửa sổ
    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        print("🔄 Đang chọn lại mô hình...")
        new_model_path = choose_model()
        if new_model_path:
            try:
                model = load_model(new_model_path)
                model_path = new_model_path
                print(f"✅ Đã load mô hình mới: {model_path}")
            except Exception as e:
                print(f"❌ Lỗi khi load mô hình mới: {e}")

cap.release()
cv2.destroyAllWindows()
