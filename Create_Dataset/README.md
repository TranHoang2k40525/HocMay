# Thư Mục Create_Dataset - Tạo Dataset từ Video

## Tổng Quan

Thư mục `Create_Dataset` chứa notebook `create_dataset.ipynb` để tạo dataset huấn luyện từ các video mẫu. Notebook này xử lý video từ thư mục `videotrain/`, trích xuất keypoints từ MediaPipe, tạo sequences và lưu thành format phù hợp cho mô hình LSTM.

## Cấu Trúc Notebook

### 1. Setup và Import (Cell 1)

**Import Libraries:**
```python
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
```

**Khởi tạo MediaPipe:**
```python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Hỗ trợ 2 tay
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
```

**Cấu hình tham số:**
```python
DATA_DIR = 'videotrain/'      # Thư mục chứa video
N_FRAMES = 30                 # Số frame trong sequence
OUTPUT_DIR = 'dataset/'       # Thư mục lưu dataset
GESTURES = [                  # 11 loại cử chỉ
    'clickchuotphai', 'clickchuottrai', 'dichuyenchuot', 'dungchuongtrinh',
    'mochorme', 'phongto', 'thunho', 'vuotlen', 'vuotphai', 'vuottrai', 'vuotxuong'
]
```

### 2. Phần 1: Detect Tay Real-Time (Cells 2-3)

**Cell 2: Mô tả chức năng**
- Mở webcam để test phát hiện tay
- Hiển thị landmarks với màu khác nhau cho 2 tay
- Vẽ endpoint (tip ngón) màu vàng
- Bbox viền mỏng, không che landmarks
- Label "Tay 1/2" trên bbox

**Cell 3: Code detect tay**
```python
# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam!")
else:
    print("Mở webcam thành công! Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Màu khác nhau cho tay
            if hand_idx == 0:
                landmark_color = (0, 255, 0)  # Xanh lá
                connection_color = (0, 255, 0)
                bbox_color = (255, 0, 0)  # Viền xanh dương
                label = "Tay 1"
            else:
                landmark_color = (0, 0, 255)  # Đỏ
                connection_color = (0, 0, 255)
                bbox_color = (0, 255, 255)  # Viền vàng
                label = "Tay 2"
            
            # Vẽ endpoint TRƯỚC (tip ngón: 4,8,12,16,20)
            h, w, _ = frame.shape
            endpoint_landmarks = [4, 8, 12, 16, 20]
            for lm_idx in endpoint_landmarks:
                lm = hand_landmarks.landmark[lm_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Vàng
            
            # Vẽ landmarks & connections
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2)
            )
            
            # Tính bbox
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Vẽ bbox SAU - Viền mỏng, không fill
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), bbox_color, 2)
            
            # Vẽ label SAU - Trên bbox
            label_pos = (x_min-20, y_min-30)
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, landmark_color, 2)
    
    cv2.imshow('Hand Detection - 2 Tay Hoàn Thiện', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Tính năng:**
- **2 tay**: Màu xanh (Tay 1), đỏ (Tay 2)
- **Endpoint**: Tip ngón tay màu vàng
- **Landmarks**: 21 điểm trên mỗi tay
- **Bbox**: Viền mỏng, không che landmarks
- **Labels**: Hiển thị "Tay 1/2"

### 3. Phần 2: Tạo Dataset (Cells 4-8)

**Cell 4: Mô tả chức năng**
- Extract keypoints cho 2 tay (mỗi tay 42 features → concat 84 features/frame)
- Nếu chỉ 1 tay, tay thứ 2 = zeros(42)
- Normalize riêng bbox/tay
- Tạo sequences N=30, pad nếu cần
- Kiểm tra 5 samples ngẫu nhiên

**Cell 5: Hàm extract keypoints**
```python
def extract_keypoints_from_frame(frame_rgb, multi_landmarks):
    """Extract keypoints cho 2 tay: Luôn concat 42*2=84 features, pad zeros nếu <2 tay."""
    all_keypoints = np.zeros(84)  # Default nếu không detect gì
    
    if not multi_landmarks:
        return all_keypoints
    
    h, w, _ = frame_rgb.shape
    tay_features = []
    
    # Luôn loop 2 tay (fixed), pad nếu thiếu
    for hand_idx in range(2):
        if hand_idx < len(multi_landmarks) and multi_landmarks[hand_idx]:
            landmarks = multi_landmarks[hand_idx]
            keypoints = []
            x_min, y_min, x_max, y_max = w, h, 0, 0
            
            # Thu thập raw keypoints (21 landmarks)
            for lm in landmarks.landmark:
                x, y = lm.x * w, lm.y * h
                keypoints.extend([x, y])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Normalize (tránh /0)
            bbox_width = max(x_max - x_min, 1)
            bbox_height = max(y_max - y_min, 1)
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            
            normalized = []
            for i in range(0, len(keypoints), 2):
                x_norm = (keypoints[i] - center_x) / bbox_width
                y_norm = (keypoints[i + 1] - center_y) / bbox_height
                normalized.extend([x_norm, y_norm])
            
            tay_features.extend(normalized)  # 42 features
        else:
            tay_features.extend(np.zeros(42).tolist())  # Pad zeros cho tay thiếu
    
    kp_array = np.array(tay_features)
    # Đảm bảo luôn 84 (nếu lỗi)
    if kp_array.shape[0] != 84:
        print(f"WARNING: Keypoints shape sai: {kp_array.shape} → Pad to 84")
        kp_array = np.pad(kp_array, (0, 84 - kp_array.shape[0]), 'constant')
    
    return kp_array  # Luôn (84,)
```

**Tính năng extract:**
- **2 tay**: Luôn tạo 84 features (42 × 2)
- **Normalize**: Theo bounding box của mỗi tay
- **Padding**: Zeros cho tay thiếu
- **Error handling**: Đảm bảo shape (84,)

**Cell 6: Hàm process video**
```python
def process_video(video_path, gesture_label):
    """Process video: Extract sequences với 2 tay, debug shapes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không mở được: {video_path}")
        return []
    
    sequences = []
    frame_keypoints = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        keypoints = extract_keypoints_from_frame(frame_rgb, results.multi_hand_landmarks)
        frame_keypoints.append(keypoints)
        frame_count += 1
        
        # Debug shape mỗi 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: keypoints shape {keypoints.shape}")
        
        if len(frame_keypoints) == N_FRAMES:
            try:
                seq_array = np.array(frame_keypoints)  # Bây giờ luôn (30,84)
                sequences.append(seq_array)
                print(f"Sequence created: shape {seq_array.shape}")
            except ValueError as e:
                print(f"ERROR tạo sequence: {e} - Skip")
                print("Last 5 shapes:", [kp.shape for kp in frame_keypoints[-5:]])
            frame_keypoints = []
    
    # Pad nếu dư frames (<30)
    if frame_keypoints:
        try:
            padded_frames = np.array(frame_keypoints)
            pad_len = N_FRAMES - len(padded_frames)
            padded = np.zeros((pad_len, 84))
            full_padded = np.vstack([padded_frames, padded])
            sequences.append(full_padded)
            print(f"Padded sequence: shape {full_padded.shape}")
        except ValueError as e:
            print(f"ERROR pad: {e} - Skip")
    
    cap.release()
    print(f"Video {os.path.basename(video_path)}: {len(sequences)} sequences (2 tay, 84 features)")
    return sequences
```

**Tính năng process:**
- **Sequence creation**: Tạo sequences 30 frames
- **Error handling**: Xử lý lỗi shape
- **Padding**: Pad frames cuối nếu <30
- **Debug info**: Hiển thị shape và progress

**Cell 7: Xử lý tất cả video**
```python
all_sequences = []
all_labels = []

# Debug thư mục
print(f"DATA_DIR: {DATA_DIR} (exists? {os.path.exists(DATA_DIR)})")
if os.path.exists(DATA_DIR):
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"Subdirs found: {subdirs[:3]}...")

for gesture in GESTURES:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(gesture_dir):
        print(f"Không tìm thấy: {gesture_dir} - Bỏ qua.")
        continue
    
    video_files = [f for f in os.listdir(gesture_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"{gesture}: Không có video hợp lệ!")
        continue
    print(f"\nProcessing {gesture} ({len(video_files)} videos)")
    
    for video_file in video_files:
        video_path = os.path.join(gesture_dir, video_file)
        seqs = process_video(video_path, gesture)
        all_sequences.extend(seqs)
        all_labels.extend([gesture] * len(seqs))

# Handle nếu rỗng
if not all_sequences:
    print("ERROR: Không có sequences! Kiểm tra video/path.")
else:
    # Numpy arrays
    X = np.array(all_sequences)  # (num_seq, 30, 84)
    y_str = np.array(all_labels)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # Lưu chuẩn cho LSTM
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'label_encoder.npy'), label_encoder.classes_)

    print(f"\nDataset hoàn thiện! X: {X.shape}, y: {y.shape} (classes: {len(np.unique(y))})")

    # Thống kê classes
    from collections import Counter
    class_counts = Counter(y_str)
    print("Số sequences/gesture:", dict(class_counts))
```

**Kết quả xử lý:**
- **Dataset**: X (203, 30, 84), y (203,)
- **Classes**: 11 loại cử chỉ
- **Files**: X.npy, y.npy, label_encoder.npy

**Cell 8: Kiểm tra samples**
```python
# Load (nếu chạy lại)
X = np.load(os.path.join(OUTPUT_DIR, 'X.npy'))
y = np.load(os.path.join(OUTPUT_DIR, 'y.npy'))
label_encoder = np.load(os.path.join(OUTPUT_DIR, 'label_encoder.npy'), allow_pickle=True)

def decode_label(encoded_y):
    return label_encoder[encoded_y]

num_samples = min(5, len(X))
random_indices = random.sample(range(len(X)), num_samples)

print("=== KIỂM TRA 5 SEQUENCES (2 TAY) ===")
for i, idx in enumerate(random_indices):
    seq = X[idx]
    label_name = decode_label(y[idx])
    
    print(f"\nSample {i+1} (Index {idx}):")
    print(f"- Hành động: {label_name}")
    print(f"- Shape: {seq.shape} (N={N_FRAMES}, 84 features = 2 tay)")
    print(f"- Mẫu keypoints frame 0 (Tay1, 5 landmarks đầu x,y):")
    tay1_sample = seq[0][:10].reshape(-1, 2)  # 5 lm x (x,y) của tay1
    print(tay1_sample)
    print("-" * 50)
```

**Kết quả kiểm tra:**
- **5 samples**: Random từ dataset
- **Shape**: (30, 84) - 30 frames, 84 features
- **Keypoints**: Normalized theo bounding box
- **Labels**: 11 loại cử chỉ

## Cấu Trúc Thư Mục Video

```
videotrain/
├── clickchuotphai/
│   └── clickchuotphai.mp4
├── clickchuottrai/
│   └── clickchuottrai.mp4
├── dichuyenchuot/
│   └── dichuyenchuot.mp4
├── dungchuongtrinh/
│   └── dungchuongtrinh.mp4
├── mochorme/
│   └── mochorme.mp4
├── phongto/
│   └── phongto.mp4
├── thunho/
│   └── thunho.mp4
├── vuotlen/
│   └── vuotlen.mp4
├── vuotphai/
│   └── vuotphai.mp4
├── vuottrai/
│   └── vuottrai.mp4
└── vuotxuong/
    └── vuotxuong.mp4
```

## Cách Chạy

### 1. Chuẩn Bị Video
1. Tạo thư mục `videotrain/`
2. Tạo 11 thư mục con cho mỗi cử chỉ
3. Đặt video mẫu vào từng thư mục
4. Đảm bảo video có format .mp4, .avi, .mov

### 2. Chạy Notebook
1. Mở `create_dataset.ipynb`
2. Chạy từng cell theo thứ tự
3. Kiểm tra kết quả trong thư mục `dataset/`

### 3. Sử Dụng Dataset
```python
# Load dataset
X = np.load('dataset/X.npy')
y = np.load('dataset/y.npy')
label_encoder = np.load('dataset/label_encoder.npy', allow_pickle=True)

# Shape: X (num_sequences, 30, 84), y (num_sequences,)
print(f"Dataset: {X.shape}, {y.shape}")
```

## Kết Quả Dataset

### 1. Thống Kê
- **Total sequences**: 203
- **Sequence length**: 30 frames
- **Features**: 84 (42 × 2 tay)
- **Classes**: 11 loại cử chỉ

### 2. Phân Bố Classes
```
clickchuotphai: 18 sequences
clickchuottrai: 20 sequences
dichuyenchuot: 17 sequences
dungchuongtrinh: 18 sequences
mochorme: 19 sequences
phongto: 21 sequences
thunho: 20 sequences
vuotlen: 18 sequences
vuotphai: 15 sequences
vuottrai: 16 sequences
vuotxuong: 16 sequences
```

### 3. Files Output
- **X.npy**: Features array (203, 30, 84)
- **y.npy**: Labels array (203,)
- **label_encoder.npy**: Label encoder classes

## Tối Ưu Hóa

### 1. Chất Lượng Video
- **Resolution**: Tối thiểu 640x480
- **FPS**: 30 FPS (phù hợp với N_FRAMES=30)
- **Duration**: 10-30 giây mỗi video
- **Lighting**: Đủ sáng, không bóng tối

### 2. Cử Chỉ Mẫu
- **Consistency**: Thực hiện cử chỉ nhất quán
- **Variety**: Nhiều góc độ, khoảng cách khác nhau
- **Quality**: Cử chỉ rõ ràng, dễ nhận diện

### 3. Xử Lý Dữ Liệu
- **Normalization**: Theo bounding box của mỗi tay
- **Padding**: Zeros cho tay thiếu
- **Error handling**: Xử lý lỗi shape và format

## Mở Rộng

### 1. Thêm Cử Chỉ Mới
1. Tạo thư mục mới trong `videotrain/`
2. Thêm tên cử chỉ vào `GESTURES` list
3. Chạy lại notebook
4. Cập nhật label encoder

### 2. Cải Thiện Chất Lượng
- **Data augmentation**: Xoay, scale, noise
- **More samples**: Thêm video mẫu
- **Quality control**: Kiểm tra sequences kém

### 3. Tối Ưu Hiệu Suất
- **Batch processing**: Xử lý nhiều video cùng lúc
- **Parallel processing**: Sử dụng multiprocessing
- **Memory optimization**: Xử lý video lớn

---

**Lưu ý**: Dataset được tạo từ video mẫu, cần đảm bảo chất lượng video và cử chỉ để có kết quả huấn luyện tốt. Kết quả cuối cùng là các file .npy được sử dụng trong notebook huấn luyện.
