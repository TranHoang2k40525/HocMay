# Thư Mục Main - Module Chính

## Tổng Quan

Thư mục `Main` chứa các module chính của hệ thống điều khiển máy tính bằng cử chỉ tay. Đây là phần core của chương trình, xử lý việc phát hiện tay, nhận diện cử chỉ và thực thi các hành động điều khiển.

## Cấu Trúc Files

### 1. Main.py - File Chạy Chính

**Chức năng:**
- Khởi tạo webcam và các module cần thiết
- Vòng lặp chính xử lý video real-time
- Điều phối giữa detection, prediction và action execution
- Hiển thị giao diện người dùng

**Luồng xử lý:**
```python
while cap.isOpened():
    # 1. Đọc frame từ webcam
    ret, frame = cap.read()
    
    # 2. Phát hiện tay và trích xuất keypoints
    results = hands.process(frame_rgb)
    keypoints, hand_centers, hand_fingers = extract_keypoints_from_frame(...)
    
    # 3. Thêm vào buffer và dự đoán cử chỉ
    sequence_buffer.append(keypoints)
    gesture_label, confidence, pred_label, gesture_type = predict_gesture(...)
    
    # 4. Thực thi hành động tương ứng
    execute_action(...)
    
    # 5. Vẽ landmarks và hiển thị
    frame = draw_hand_landmarks(...)
    cv2.imshow('Gesture Recognition', frame)
```

**Tham số quan trọng:**
- `sequence_buffer`: Buffer 30 frames để dự đoán
- `previous_centers`: Lưu vị trí tay trước đó
- `previous_mouse_pos`: Vị trí chuột trước đó
- `last_discrete_time`: Thời gian cử chỉ discrete cuối

### 2. Detection.py - Phát Hiện Tay

**Chức năng:**
- Cấu hình và khởi tạo MediaPipe Hands
- Trích xuất keypoints từ frame video
- Đếm số ngón tay duỗi
- Vẽ landmarks và bounding box
- Hiển thị thông tin debug

**Các hàm chính:**

#### `extract_keypoints_from_frame(frame_rgb, multi_landmarks)`
```python
# Trích xuất 84 features (42 × 2 tay)
# Normalize theo bounding box
# Trả về: keypoints, hand_centers, hand_fingers
```

#### `count_extended_fingers(landmarks, h, w)`
```python
# Đếm ngón tay duỗi (bỏ thumb)
# Sử dụng tip và PIP landmarks
# Trả về: số ngón duỗi (0-4)
```

#### `draw_hand_landmarks(frame, results, ...)`
```python
# Vẽ landmarks với màu khác nhau cho 2 tay
# Vẽ bounding box và labels
# Hiển thị thông tin debug
```

**Cấu hình MediaPipe:**
```python
MP_HANDS_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 2,                    # Hỗ trợ 2 tay
    'min_detection_confidence': 0.7,       # Ngưỡng phát hiện
    'min_tracking_confidence': 0.5         # Ngưỡng theo dõi
}
```

### 3. Model.py - Xử Lý Mô Hình

**Chức năng:**
- Load mô hình LSTM đã huấn luyện
- Dự đoán cử chỉ từ sequence buffer
- Xử lý confidence threshold
- Mapping gesture types

**Các hàm chính:**

#### `load_gesture_model()`
```python
# Load file gesture_lstm_model.h5
# Load label encoder
# Trả về: model, label_encoder
```

#### `predict_gesture(model, label_encoder, sequence_buffer)`
```python
# Kiểm tra buffer đủ 30 frames
# Reshape input: (1, 30, 84)
# Dự đoán và áp dụng confidence threshold
# Trả về: gesture_label, confidence, pred_label, gesture_type
```

**Cấu hình mô hình:**
```python
N_FRAMES = 30          # Số frame trong sequence
FEATURES = 84          # 42 features × 2 tay
CONF_THRESHOLD = 0.5   # Ngưỡng tin cậy
```

**11 loại cử chỉ:**
```python
LABEL_ENCODER = [
    'clickchuotphai', 'clickchuottrai', 'dichuyenchuot', 'dungchuongtrinh',
    'mochorme', 'phongto', 'thunho', 'vuotlen', 'vuotphai', 'vuottrai', 'vuotxuong'
]
```

**Phân loại cử chỉ:**
```python
GESTURE_TYPES = {
    'dichuyenchuot': 'continuous',    # Di chuyển chuột
    'vuotlen': 'continuous',          # Cuộn lên
    'vuotxuong': 'continuous',        # Cuộn xuống
    'vuotphai': 'continuous',         # Tab tiếp
    'vuottrai': 'continuous',         # Tab trước
    # Các cử chỉ khác là 'discrete'
}
```

### 4. Actions.py - Thực Thi Hành Động

**Chức năng:**
- Mapping cử chỉ thành hành động cụ thể
- Xử lý di chuyển chuột với smoothing
- Thực thi các phím tắt và lệnh hệ thống
- Xử lý cử chỉ continuous và discrete

**Các hàm chính:**

#### `execute_mouse_action(curr_x_norm, curr_y_norm, previous_mouse_pos, hand_idx)`
```python
# Chuyển đổi tọa độ normalized sang screen coordinates
# Áp dụng smoothing để di chuyển mượt mà
# Cập nhật vị trí chuột
```

#### `execute_scroll_up/down(delta_y, num_fingers)`
```python
# Kiểm tra có đủ 2 ngón tay
# Tính toán scroll amount dựa trên delta_y
# Thực thi pyautogui.scroll()
```

#### `execute_tab_next/prev(delta_x)`
```python
# Kiểm tra delta_x > threshold
# Thực thi Ctrl+Tab hoặc Ctrl+Shift+Tab
```

**Cấu hình hành động:**
```python
SCROLL_SENSITIVITY = 3.0    # Độ nhạy cuộn
SMOOTH_ALPHA = 0.5          # Độ mượt di chuyển chuột
TAB_THRESHOLD = 0.05        # Ngưỡng chuyển tab
DISCRETE_DELAY = 0.2        # Delay cho cử chỉ discrete
```

**Action mapping:**
```python
action_map = {
    'clickchuotphai': execute_right_click,
    'clickchuottrai': execute_left_click,
    'dungchuongtrinh': execute_stop_program,
    'mochorme': execute_open_chrome,
    'phongto': execute_zoom_in,
    'thunho': execute_zoom_out,
    'vuotlen': execute_scroll_up,
    'vuotxuong': execute_scroll_down,
    'vuotphai': execute_tab_next,
    'vuottrai': execute_tab_prev
}
```

## Cách Chạy

### 1. Chuẩn Bị
```bash
# Đảm bảo có file model
ls ../gesture_lstm_model.h5

# Cài đặt dependencies
pip install mediapipe tensorflow opencv-python pyautogui
```

### 2. Chạy Chương Trình
```bash
cd Main
python Main.py
```

### 3. Sử Dụng
- **Mở webcam**: Chương trình tự động mở webcam
- **Thực hiện cử chỉ**: Giơ tay trước webcam
- **Xem kết quả**: Cử chỉ được nhận diện hiển thị trên màn hình
- **Thoát**: Nhấn 'q'

## Debug và Monitoring

### 1. Thông Tin Debug
```python
# Hiển thị trên console
print(f"*** DETECTED: {pred_label} (Conf: {confidence:.2f}) | Type: {gesture_type} ***")
print(f"FPS: {fps:.1f}")
print(f"Mouse zero-delay to ({curr_x_screen}, {curr_y_screen})")
```

### 2. Visual Feedback
- **Landmarks**: Vẽ 21 điểm trên mỗi tay
- **Bounding box**: Viền bao quanh tay
- **Labels**: Hiển thị cử chỉ và confidence
- **Colors**: Tay 1 (xanh), Tay 2 (đỏ)
- **Arrows**: Hiển thị hướng di chuyển

### 3. Performance Monitoring
- **FPS counter**: Hiển thị tốc độ xử lý
- **Buffer status**: Hiển thị trạng thái buffer
- **Confidence**: Hiển thị độ tin cậy dự đoán

## Xử Lý Lỗi

### 1. Lỗi Webcam
```python
if not cap.isOpened():
    print("Không mở được webcam!")
    exit(1)
```

### 2. Lỗi Model
```python
if not os.path.exists(MODEL_PATH):
    print(f"Không tìm model tại {MODEL_PATH}!")
    exit(1)
```

### 3. Lỗi Buffer
```python
if len(sequence_buffer) != N_FRAMES:
    return "Chờ buffer...", 0.0, "No action", "discrete"
```

## Tối Ưu Hóa

### 1. Hiệu Suất
- **Buffer size**: 30 frames (cân bằng độ chính xác/tốc độ)
- **Confidence threshold**: 0.5 (giảm false positive)
- **Smoothing**: Alpha = 0.5 (mượt mà di chuyển chuột)

### 2. Độ Chính Xác
- **MediaPipe config**: Tối ưu cho real-time
- **Gesture types**: Phân biệt continuous/discrete
- **Action mapping**: Mapping chính xác cử chỉ → hành động

### 3. Trải Nghiệm Người Dùng
- **Visual feedback**: Hiển thị đầy đủ thông tin
- **Color coding**: Phân biệt 2 tay
- **Real-time**: Xử lý mượt mà, không lag

## Mở Rộng

### 1. Thêm Cử Chỉ Mới
1. Cập nhật `LABEL_ENCODER` trong `Model.py`
2. Thêm action mapping trong `Actions.py`
3. Cập nhật `GESTURE_TYPES` nếu cần
4. Huấn luyện lại mô hình

### 2. Cải Thiện Hiệu Suất
- **Multi-threading**: Xử lý song song
- **GPU acceleration**: Sử dụng CUDA
- **Model optimization**: Quantization, pruning

### 3. Tích Hợp Ứng Dụng
- **API endpoints**: REST API cho remote control
- **Plugin system**: Hỗ trợ plugin tùy chỉnh
- **Configuration**: File config cho settings

---

**Lưu ý**: Đây là module core của hệ thống, cần được chạy với đầy đủ dependencies và file model đã huấn luyện.
