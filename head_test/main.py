import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
from scipy.linalg import inv

# ====================== 全局参数 ======================
# 卡尔曼滤波参数
KF_A = np.array([[1, 1, 0.5],
                  [0, 1, 1],
                  [0, 0, 1]])
KF_H = np.array([[1, 0, 0]])
KF_Q = np.diag([0.01, 0.1, 0.5])
KF_R = np.array([[10]])

# 左右车道线卡尔曼滤波器
left_kf_x = np.array([[0], [0], [0]])
left_kf_P = np.eye(3)
right_kf_x = np.array([[0], [0], [0]])
right_kf_P = np.eye(3)

# ======================================================
# 1. 颜色空间融合（HSL + LAB）
# ======================================================
def select_color(image):
    hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # 白色：HSL高亮度 + LAB高亮度
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsl, lower_white, upper_white)

    # 黄色：HSL黄区间 + LAB b通道
    lower_yellow = np.array([10, 0, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsl, lower_yellow, upper_yellow)

    # LAB增强白色
    l_channel = lab[:, :, 0]
    _, lab_white = cv2.threshold(l_channel, 210, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    mask = cv2.bitwise_or(mask, lab_white)
    return cv2.bitwise_and(image, image, mask=mask)

# ======================================================
# 2. 自适应灰度+模糊+Canny
# ======================================================
def adaptive_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 自适应直方图均衡化（抗光照）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def adaptive_blur(gray):
    # 根据亮度动态核大小
    brightness = np.mean(gray)
    ksize = 5 if brightness > 120 else 9
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)

def adaptive_canny(blurred):
    # 自动计算高低阈值
    median = np.median(blurred)
    low = int(max(0, 0.4 * median))
    high = int(min(255, 1.4 * median))
    return cv2.Canny(blurred, low, high)

# ======================================================
# 3. 动态ROI（梯形）
# ======================================================
def select_region(image):
    h, w = image.shape
    # 动态梯形顶点
    top = int(h * 0.6)
    bottom_left = (int(w * 0.1), h)
    bottom_right = (int(w * 0.9), h)
    top_left = (int(w * 0.4), top)
    top_right = (int(w * 0.6), top)

    mask = np.zeros_like(image)
    pts = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(image, mask)

# ======================================================
# 4. 概率霍夫变换 + 线段过滤
# ======================================================
def hough_lines(roi_img):
    lines = cv2.HoughLinesP(
        roi_img,
        rho=2,
        theta=np.pi/180,
        threshold=30,
        minLineLength=30,
        maxLineGap=10
    )
    if lines is None:
        return []
    return lines

# ======================================================
# 5. 斜率加权拟合 + 卡尔曼滤波跟踪
# ======================================================
def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        # 过滤过平/过陡
        if abs(slope) < 0.4 or abs(slope) > 2.0:
            continue

        if slope < 0:
            left_lines.append((slope, intercept))
            left_weights.append(length)
        else:
            right_lines.append((slope, intercept))
            right_weights.append(length)

    # 加权平均
    left_lane = None
    if left_weights:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
    right_lane = None
    if right_weights:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)

    return left_lane, right_lane

def kalman_update(line, x, P):
    if line is None:
        return x, P
    slope, intercept = line
    z = np.array([[slope]])

    # 预测
    x_pred = KF_A @ x
    P_pred = KF_A @ P @ KF_A.T + KF_Q

    # 更新
    y = z - KF_H @ x_pred
    S = KF_H @ P_pred @ KF_H.T + KF_R
    K = P_pred @ KF_H.T @ inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(3) - K @ KF_H) @ P_pred

    return x_new, P_new

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))

# ======================================================
# 6. 曲率 + 偏离距离计算
# ======================================================
def calculate_curvature_and_offset(left_line, right_line, h, w):
    if left_line is None or right_line is None:
        return 0, 0

    # 取底部点
    y = h
    x_left = left_line[0][0]
    x_right = right_line[0][0]
    lane_width = x_right - x_left
    if lane_width <= 0:
        return 0, 0

    # 像素到米（假设3.7米车道宽）
    xm_per_pix = 3.7 / lane_width
    ym_per_pix = 3.0 / h

    # 二次拟合
    left_pts = np.array([[left_line[0][0], left_line[0][1]],
                           [left_line[1][0], left_line[1][1]]], dtype=np.float32)
    right_pts = np.array([[right_line[0][0], right_line[0][1]],
                           [right_line[1][0], right_line[1][1]]], dtype=np.float32)
    pts = np.vstack((left_pts, right_pts))
    fit = np.polyfit(pts[:, 1], pts[:, 0], 2)

    # 曲率
    curverad = ((1 + (2 * fit[0] * y * ym_per_pix + fit[1])**2)**1.5) / abs(2 * fit[0])
    # 偏移
    center = w / 2
    lane_center = (x_left + x_right) / 2
    offset = (center - lane_center) * xm_per_pix

    return curverad, offset

# ======================================================
# 7. 绘制车道线 + 状态信息
# ======================================================
def draw_lane_info(image, left_line, right_line, curvature, offset):
    overlay = np.zeros_like(image)
    if left_line:
        cv2.line(overlay, left_line[0], left_line[1], (0,255,0), 20)
    if right_line:
        cv2.line(overlay, right_line[0], right_line[1], (0,255,0), 20)
    combined = cv2.addWeighted(image, 1.0, overlay, 0.6, 0)

    # 文字信息
    info1 = f"Curvature: {curvature:.1f} m"
    info2 = f"Offset: {offset:.2f} m {'left' if offset>0 else 'right'}"
    cv2.putText(combined, info1, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(combined, info2, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # 偏离告警
    if abs(offset) > 0.5:
        cv2.putText(combined, "WARNING: Lane Departure!", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return combined

# ======================================================
# 8. 主处理流水线
# ======================================================
def LaneLine_process(image):
    global left_kf_x, left_kf_P, right_kf_x, right_kf_P
    h, w = image.shape[:2]

    # 1. 颜色筛选
    color_masked = select_color(image)
    # 2. 自适应灰度
    gray = adaptive_gray(color_masked)
    # 3. 自适应模糊
    blurred = adaptive_blur(gray)
    # 4. 自适应Canny
    edges = adaptive_canny(blurred)
    # 5. ROI
    roi = select_region(edges)
    # 6. 霍夫线
    lines = hough_lines(roi)
    # 7. 拟合
    left_lane, right_lane = average_slope_intercept(lines)
    # 8. 卡尔曼滤波
    left_kf_x, left_kf_P = kalman_update(left_lane, left_kf_x, left_kf_P)
    right_kf_x, right_kf_P = kalman_update(right_lane, right_kf_x, right_kf_P)
    left_filt = (left_kf_x[0,0], left_kf_x[1,0]) if left_lane else None
    right_filt = (right_kf_x[0,0], right_kf_x[1,0]) if right_lane else None
    # 9. 生成线段
    y1, y2 = h, h * 0.6
    left_line = make_line_points(y1, y2, left_filt)
    right_line = make_line_points(y1, y2, right_filt)
    # 10. 曲率+偏移
    curvature, offset = calculate_curvature_and_offset(left_line, right_line, h, w)
    # 11. 绘制
    result = draw_lane_info(image, left_line, right_line, curvature, offset)
    return result

# ======================================================
# 9. 运行
# ======================================================
if __name__ == '__main__':
    video_input_path = r"head_test/test_videos/challenge.mp4"
    video_output_path = "video_1_xlt.mp4"

    print(f"正在处理视频: {video_input_path} ...")
    clip = VideoFileClip(video_input_path)
    out_clip = clip.image_transform(LaneLine_process)
    out_clip.write_videofile(video_output_path, audio=True)
    print("视频处理完成！")