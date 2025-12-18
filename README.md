# Emotion Recognition using MobileNetV2

## Mô tả dự án

Hệ thống nhận diện cảm xúc khuôn mặt thời gian thực sử dụng Deep Learning với kiến trúc MobileNetV2. Dự án triển khai giải pháp AI để phân loại 7 loại cảm xúc cơ bản (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) từ hình ảnh khuôn mặt.

**Công nghệ sử dụng:**
- **Deep Learning Framework:** TensorFlow/Keras với MobileNetV2 (transfer learning)
- **Computer Vision:** OpenCV cho xử lý ảnh và phát hiện khuôn mặt
- **GUI Development:** Tkinter/CustomTkinter cho giao diện người dùng
- **Data Processing:** NumPy, Pandas, Scikit-learn

**Tính năng chính:**
- Nhận diện cảm xúc thời gian thực từ webcam với độ chính xác cao
- Giao diện người dùng thân thiện, dễ sử dụng
- Lưu trữ và phân loại khuôn mặt theo cảm xúc
- Hỗ trợ nhiều phương pháp tiền xử lý ảnh (CLAHE, Edge Detection, Median Filter)
- Xuất báo cáo phân tích với confusion matrix và classification report

**Kết quả:**
- Độ chính xác trên tập test: ~70-75%
- Tốc độ xử lý: Real-time (>20 FPS)
- Model size: ~160MB (MobileNetV2 optimized)

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
