# Emotion Recognition using MobileNetV2

## Setup
DÙNG PYTHON 3.10
venv\Scripts\activate

1. Cài đặt các thư viện yêu cầu:
   pip install -r requirements.txt

2. Huấn luyện mô hình với `train.py`:
   python train.py

3. Nhận diện cảm xúc từ webcam với `detect_emotion.py`:
   python real_time_detection/detect_emotion.py

4. Sử dụng giao diện GUI với `app.py`:
   python gui/app.py

## Các mô-đun:

- **train.py**: Huấn luyện mô hình MobileNetV2 cho nhận diện cảm xúc.
- **detect_emotion.py**: Nhận diện cảm xúc từ webcam.
- **app.py**: Giao diện người dùng để nhận diện cảm xúc từ ảnh tải lên.
