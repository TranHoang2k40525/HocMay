# actions.py - Các hàm execute actions (gộp config: SCROLL_SENSITIVITY, SMOOTH_ALPHA, etc.)

import pyautogui
import time
from Model import GESTURE_TYPES  # Import GESTURE_TYPES từ model.py
# Gộp config actions
SCROLL_AMOUNT = 100  
SMOOTH_ALPHA = 0.5  
DISCRETE_COOLDOWN = 1.0 
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# Tracking last execution time cho từng gesture (cooldown)
last_execution_times = {}

def execute_right_click():
    pyautogui.rightClick()
    print("Executed: Right click!")
    return False  # Không dừng chương trình

def execute_left_click():
    pyautogui.leftClick()
    print("Executed: Left click!")
    return False  # Không dừng chương trình

def execute_stop_program():
    print("Executed: Dừng chương trình! (Thoát)")
    return False

def execute_open_app():
    """Mở app, không dừng chương trình, nhưng chống spam bằng cooldown riêng."""
    global last_execution_times
    app_label = 'moapp'
    current_time = time.time()
    # Cooldown riêng cho mở app (5 giây)
    APP_COOLDOWN = 5.0
    if app_label in last_execution_times:
        time_since_last = current_time - last_execution_times[app_label]
        if time_since_last < APP_COOLDOWN:
            print(f"Skip mở app: Cooldown còn {APP_COOLDOWN - time_since_last:.2f}s")
            return False
    pyautogui.hotkey('win', 'r')
    time.sleep(0.5)  # Đợi Run dialog mở
    pyautogui.write('"C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Coc Coc.lnk"')
    pyautogui.press('enter')
    print("Executed: Mở Coc Coc!")
    last_execution_times[app_label] = current_time
    return False  # Không dừng chương trình

def execute_zoom_in():
    pyautogui.hotkey('ctrl', '+')
    print("Executed: Phóng to!")
    return False

def execute_zoom_out():
    pyautogui.hotkey('ctrl', '-')
    print("Executed: Thu nhỏ!")
    return False

def execute_tab_next():
    """Chuyển tab tiếp theo (Ctrl+Tab)."""
    pyautogui.hotkey('ctrl', 'tab')
    print("Executed: Tab next (Ctrl+Tab)")
    return False

def execute_tab_prev():
    """Chuyển tab trước đó (Ctrl+Shift+Tab)."""
    pyautogui.hotkey('ctrl', 'shift', 'tab')
    print("Executed: Tab prev (Ctrl+Shift+Tab)")
    return False

def execute_scroll_up():
    """Scroll LÊN trong tab/trang hiện tại (như vuốt lên trên touchpad)."""
    # Thực hiện scroll nhiều lần cho mượt
    for _ in range(3):
        pyautogui.scroll(SCROLL_AMOUNT)
        time.sleep(0.01)
    print(f"Executed: Scroll UP {SCROLL_AMOUNT * 3} (vuốt lên trong trang)")
    return False

def execute_scroll_down():
    """Scroll XUỐNG trong tab/trang hiện tại (như vuốt xuống trên touchpad)."""
    # Thực hiện scroll nhiều lần cho mượt
    for _ in range(3):
        pyautogui.scroll(-SCROLL_AMOUNT)
        time.sleep(0.01)
    print(f"Executed: Scroll DOWN {SCROLL_AMOUNT * 3} (vuốt xuống trong trang)")
    return False

def execute_mouse_action(curr_x_norm, curr_y_norm, previous_mouse_pos, hand_idx):
    """
    Xử lý di chuyển mouse (zero-delay smooth).
    """
    screen_w, screen_h = pyautogui.size()
    curr_x_screen = int((curr_x_norm + 1) * screen_w / 2)
    curr_y_screen = int((curr_y_norm + 1) * screen_h / 2)
    if previous_mouse_pos[hand_idx] is not None:
        prev_x, prev_y = previous_mouse_pos[hand_idx]
        smooth_x = int(prev_x + SMOOTH_ALPHA * (curr_x_screen - prev_x))
        smooth_y = int(prev_y + SMOOTH_ALPHA * (curr_y_screen - prev_y))
        pyautogui.moveTo(smooth_x, smooth_y)
    else:
        pyautogui.moveTo(curr_x_screen, curr_y_screen)
    previous_mouse_pos[hand_idx] = (curr_x_screen, curr_y_screen)
    print(f"Mouse zero-delay to ({curr_x_screen}, {curr_y_screen})")

def get_action_func(pred_label):
    """
    Trả về function tương ứng từ pred_label.
    """
    action_map = {
        'clickchuotphai': execute_right_click,
        'clickchuottrai': execute_left_click,
        'dungchuongtrinh': execute_stop_program,
        'moapp': execute_open_app,
        'phongto': execute_zoom_in,
        'thunho': execute_zoom_out,
        'vuotlen': execute_scroll_up,
        'vuotxuong': execute_scroll_down,
        'vuotphai': execute_tab_next,
        'vuottrai': execute_tab_prev
    }
    return action_map.get(pred_label)

def execute_action(execute_func, pred_label, current_time):
    """
    Execute action với cooldown (TẤT CẢ là discrete trừ dichuyenchuot).
    Returns: should_stop (bool)
    """
    global last_execution_times
    
    if pred_label in last_execution_times:
        time_since_last = current_time - last_execution_times[pred_label]
        if time_since_last < DISCRETE_COOLDOWN:
            return False
    
    # Execute action
    should_stop = execute_func()
    
    # Cập nhật thời gian execute cuối
    last_execution_times[pred_label] = current_time
    
    # Return True nếu cần dừng chương trình
    return should_stop if should_stop else False