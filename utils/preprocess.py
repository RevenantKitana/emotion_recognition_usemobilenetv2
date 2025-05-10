import cv2
import numpy as np

def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))  # Sử dụng img_size cho cả chiều rộng và cao
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img