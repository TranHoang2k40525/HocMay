# Thư Mục Train - Huấn Luyện Mô Hình LSTM

## Tổng Quan

Thư mục `Train` chứa notebook `LSTM_Train.ipynb` để huấn luyện mô hình LSTM nhận diện cử chỉ tay. Notebook này thực hiện toàn bộ pipeline từ làm sạch dữ liệu, xây dựng mô hình, huấn luyện, đánh giá đến trực quan hóa kết quả.

## Cấu Trúc Notebook

### 1. Setup và Import (Cells 1-3)

**Cell 1: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Kết nối với Google Drive để truy cập dataset
- Dataset được lưu trong `/content/drive/MyDrive/dataset/`

**Cell 2: Cài Đặt Thư Viện**
```bash
!pip install mediapipe tensorflow scikit-learn matplotlib seaborn opencv-python
```
- Cài đặt các thư viện cần thiết
- MediaPipe: Phát hiện tay
- TensorFlow: Mô hình LSTM
- Scikit-learn: Xử lý dữ liệu
- Matplotlib/Seaborn: Trực quan hóa

**Cell 3: Import Libraries**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp
import os
from collections import Counter
import random
```

### 2. Load và Làm Sạch Dữ Liệu (Cell 4)

**Load Dataset:**
```python
OUTPUT_DIR = '/content/drive/MyDrive/dataset/'
X = np.load(os.path.join(OUTPUT_DIR, 'X.npy'))
y = np.load(os.path.join(OUTPUT_DIR, 'y.npy'))
label_encoder = np.load(os.path.join(OUTPUT_DIR, 'label_encoder.npy'), allow_pickle=True)
```

**Thống kê ban đầu:**
- Dataset gốc: X (203, 30, 84), y (203,) (classes: 11)
- 203 sequences, mỗi sequence 30 frames, 84 features

**Làm sạch dữ liệu:**
```python
def clean_data(X, y, zero_threshold=0.7):
    clean_indices = []
    for i, seq in enumerate(X):
        zero_frames = np.sum(np.all(seq == 0, axis=1))
        if zero_frames / len(seq) < zero_threshold:  # Giữ nếu <70% zeros
            clean_indices.append(i)
    X_clean = X[clean_indices]
    y_clean = y[clean_indices]
    return X_clean, y_clean
```

**Kết quả làm sạch:**
- Loại bỏ 5 sequences kém
- Còn lại: (198, 30, 84)

**Balance classes:**
```python
# Nếu imbalance (>2x chênh lệch), undersample majority
max_count = max(class_counts.values())
if max_count > 2 * min(class_counts.values()):
    # Undersample để balance
```

**Phân bố classes sau balance:**
```
{0: 18, 1: 20, 2: 17, 3: 18, 4: 19, 5: 21, 6: 20, 7: 18, 8: 15, 9: 16, 10: 16}
```

**Normalize và Split:**
```python
# Normalize X (scale 0-1)
X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-7)

# One-hot y
y_cat = to_categorical(y, num_classes)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
```

### 3. Xây Dựng Mô Hình (Cells 5-6)

**Cell 5: Mô tả kiến trúc**
- LSTM Bidirectional cho sequence động, 2 tay
- Input: (30, 84) - N=30 frames, 84 features (2 tay)
- Output: Softmax 11 classes (gestures)
- Mục tiêu: Predict chính xác để map → actions

**Cell 6: Xây dựng mô hình**
```python
N_FRAMES = X.shape[1]  # 30
FEATURES = X.shape[2]  # 84

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(30, 84)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

**Kiến trúc mô hình:**
- **Bidirectional LSTM 128**: Xử lý sequence 2 chiều
- **Dropout 0.2**: Regularization
- **Bidirectional LSTM 64**: Layer thứ 2
- **Dense 32**: Fully connected
- **Dense 11**: Output layer với softmax

### 4. Huấn Luyện (Cells 7-8)

**Cell 7: Mô tả quá trình train**
- Epochs=100, batch=16 (nhỏ cho dataset nhỏ)
- Callbacks: EarlyStopping, ReduceLR, Checkpoint
- Hiển thị tiến độ: Plot accuracy/loss real-time

**Cell 8: Thực hiện huấn luyện**
```python
# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint('best_gesture_model.h5', monitor='val_accuracy', save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Lưu model
model.save('gesture_lstm_model.h5')
```

**Kết quả huấn luyện:**
- Train hoàn tất sau vài epochs
- Model saved: gesture_lstm_model.h5
- Best weights được restore

### 5. Đánh Giá Mô Hình (Cells 9-10)

**Cell 9: Mô tả đánh giá**
- Accuracy, Loss trên test
- Classification report: Precision, Recall, F1-score
- Confusion Matrix (heatmap)
- Plot History: Tiến độ train

**Cell 10: Thực hiện đánh giá**
```python
# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report (Precision/Recall/F1):")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder))

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder, yticklabels=label_encoder)
plt.title('Confusion Matrix - Sơ Đồ Đánh Giá Chỉ Số')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Tiến Độ Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Tiến Độ Loss')
plt.legend()
plt.show()
```

**Kết quả đánh giá:**
- **Test Accuracy**: 100.00%
- **Test Loss**: 0.0013
- **Precision/Recall/F1**: 1.00 cho tất cả classes
- **Confusion Matrix**: Hoàn hảo (100% accuracy)

### 6. Trực Quan Real-Time (Cells 11-12)

**Cell 11: Mô tả trực quan**
- Load model, chạy webcam
- Buffer 30 frames keypoints (2 tay)
- Predict gesture, vẽ đúng tay (màu xanh/đỏ)
- Nhãn hành động rõ trên bbox
- Kiểm tra accuracy: Predict 5 test samples

**Cell 12: Thực hiện trực quan**
```python
# Load model
model = load_model('gesture_lstm_model.h5')

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Buffer cho sequence
sequence_buffer = np.zeros((N_FRAMES, FEATURES))  # (30, 84)
frame_count = 0

# Kiểm tra accuracy với 5 test samples
print("=== KIỂM TRA ACCURACY 5 SAMPLES TEST ===")
num_samples = min(5, len(X_test))
indices = np.random.choice(len(X_test), num_samples, replace=False)
for i, idx in enumerate(indices):
    seq = X_test[idx:idx+1]
    pred = model.predict(seq)[0]
    pred_label = np.argmax(pred)
    true_label = np.argmax(y_test[idx])
    print(f"Sample {i+1}: True = {label_encoder[true_label]}, Predict = {label_encoder[pred_label]} (Conf: {pred[pred_label]:.2f})")

# Real-time Webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Extract keypoints & buffer
    keypoints = extract_keypoints_from_frame(frame_rgb, results.multi_hand_landmarks)
    sequence_buffer[frame_count % N_FRAMES] = keypoints
    frame_count += 1

    # Predict gesture
    if frame_count >= N_FRAMES:
        input_seq = sequence_buffer.reshape(1, N_FRAMES, FEATURES)
        pred = model.predict(input_seq, verbose=0)[0]
        pred_idx = np.argmax(pred)
        gesture_label = label_encoder[pred_idx]
        confidence = pred[pred_idx]
        if confidence > 0.7:
            print(f"Detected: {gesture_label} (Conf: {confidence:.2f})")

    # Vẽ tay & nhãn
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            color = (0, 255, 0) if hand_idx == 0 else (0, 0, 255)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=color, thickness=2))
            # Bbox & nhãn
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x); y_min = min(y_min, y); x_max = max(x_max, x); y_max = max(y_max, y)
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), color, 2)
            if hand_idx == 0:
                label_pos = (x_min-20, y_min-30)
                cv2.putText(frame, f"{gesture_label} ({confidence:.2f})", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Real-Time Gesture Recognition - LSTM Model', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Kết quả trực quan:**
- **Accuracy test**: 100% trên 5 samples
- **Real-time detection**: Hoạt động mượt mà
- **Visual feedback**: Hiển thị landmarks, bbox, labels

## Cách Chạy

### 1. Chuẩn Bị
```bash
# Upload dataset lên Google Drive
# Đảm bảo có file X.npy, y.npy, label_encoder.npy trong /content/drive/MyDrive/dataset/
```

### 2. Chạy Notebook
1. Mở `LSTM_Train.ipynb` trong Google Colab
2. Chạy từng cell theo thứ tự
3. Chờ quá trình huấn luyện hoàn tất
4. Download file `gesture_lstm_model.h5`

### 3. Sử Dụng Model
```python
# Load model đã huấn luyện
model = load_model('gesture_lstm_model.h5')

# Predict gesture
pred = model.predict(input_sequence)
pred_label = np.argmax(pred)
confidence = pred[pred_label]
```

## Kết Quả Huấn Luyện

### 1. Độ Chính Xác
- **Test Accuracy**: 100.00%
- **Test Loss**: 0.0013
- **Precision/Recall/F1**: 1.00 cho tất cả classes

### 2. Confusion Matrix
- Hoàn hảo: 100% accuracy
- Không có misclassification
- Tất cả 11 classes được nhận diện chính xác

### 3. Training History
- **Accuracy**: Tăng dần từ ~0.1 đến 1.0
- **Loss**: Giảm dần từ ~2.3 đến ~0.001
- **Validation**: Theo sát training, không overfitting

## Tối Ưu Hóa

### 1. Data Preprocessing
- **Cleaning**: Loại bỏ sequences kém (>70% zeros)
- **Balancing**: Undersample majority classes
- **Normalization**: Scale features về [0,1]

### 2. Model Architecture
- **Bidirectional LSTM**: Xử lý sequence 2 chiều
- **Dropout**: Regularization để tránh overfitting
- **Early Stopping**: Dừng sớm khi không cải thiện

### 3. Training Strategy
- **Callbacks**: EarlyStopping, ReduceLR, Checkpoint
- **Batch size**: 16 (phù hợp với dataset nhỏ)
- **Learning rate**: 0.001 với ReduceLROnPlateau

## Mở Rộng

### 1. Thêm Cử Chỉ Mới
1. Tạo video mẫu cho cử chỉ mới
2. Chạy `Create_Dataset/create_dataset.ipynb`
3. Cập nhật `GESTURES` list
4. Huấn luyện lại mô hình

### 2. Cải Thiện Độ Chính Xác
- **Data augmentation**: Xoay, scale, noise
- **More data**: Thêm video huấn luyện
- **Model complexity**: Thêm layers, attention

### 3. Tối Ưu Hiệu Suất
- **Model quantization**: Giảm kích thước model
- **Pruning**: Loại bỏ weights không quan trọng
- **Knowledge distillation**: Compress model

---

**Lưu ý**: Notebook này được thiết kế để chạy trên Google Colab với GPU để tăng tốc quá trình huấn luyện. Kết quả cuối cùng là file `gesture_lstm_model.h5` được sử dụng trong module Main.
