import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip

# ==========================================
# 1. 颜色提取与基础图像处理
# ==========================================

def convert_hsl(image):
    """将图像转换为 HSL 颜色空间"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_hsl_white_yellow(image):
    """使用 HSL 颜色空间对图片进行特定颜色过滤（提取白线和黄线）"""
    hsl = convert_hsl(image)
    
    # 提取白色的掩膜
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hsl, lower_white, upper_white)
    
    # 提取黄色的掩膜
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(hsl, lower_yellow, upper_yellow)
    
    # 合并二值图
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    # 二值图与原图进行按位与，得到黄线与白线
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def convert_gray_scale(image):
    """灰度化处理"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smooothing(image, ksize=15):
    """高斯滤波去噪"""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def detect_edges(image, low_thresh=50, high_thresh=150):
    """Canny 边缘检测"""
    return cv2.Canny(image, low_thresh, high_thresh)

# ==========================================
# 2. ROI 感兴趣区域提取
# ==========================================

def filter_region(image, vertices):
    """多边形填充 + 按位与过滤（掩膜处理）"""
    mask = np.zeros_like(image) 
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])
    return cv2.bitwise_and(image, mask)

def select_region(image):
    """定义感兴趣区域 (ROI) 的顶点并提取"""
    rows, cols = image.shape[0:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

# ==========================================
# 3. 霍夫直线检测与车道线拟合计算
# ==========================================

def hough_lines(image):
    """霍夫直线检测"""
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

def average_slope_intercept(lines):
    """车道线合理性判断与斜率计算：剔除误差较大的点，并区分左右车道"""
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    
    if lines is None:
        return None, None
        
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1: # 忽略垂直线
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            
            # y 轴方向在图像中是向下的，因此斜率为负代表左车道线，为正代表右车道线
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
                
    # 根据线段长度加权平均
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    """根据直线的斜率和截距，计算在图像上的具体像素坐标端点"""
    if line is None:
        return None
    slope, intercept = line
    
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """延长并确定最终车道线的端点"""
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0] # 图像底部
    y2 = y1 * 0.6       # 延伸至 ROI 顶部
    
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=(255, 0, 0), thickness=20):
    """在图像上绘制最终拟合出的粗实线"""
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, line[0], line[1], color, thickness)
    
    # 将车道线与原图叠加
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0)

# ==========================================
# 4. 视频逐帧处理流水线
# ==========================================

def LaneLine_process(image):
    """车道线检测核心处理流程"""
    test_image = image
    
    # 1. 提取黄白部分
    white_yello_images = select_hsl_white_yellow(test_image)
    # 2. 灰度化
    gray_images = convert_gray_scale(white_yello_images)
    # 3. 滤波去噪
    blured_images = apply_smooothing(gray_images)
    # 4. 边缘检测
    edge_images = detect_edges(blured_images)
    # 5. 获取感兴趣区域
    roi_images = select_region(edge_images)
    # 6. 获得潜在直线段
    list_of_lines = hough_lines(roi_images)
    # 7. 拟合并绘制车道线
    lane_images = draw_lane_lines(test_image, lane_lines(test_image, list_of_lines))
    
    return lane_images


if __name__ == '__main__':

    video_input_path = r"head _test/test_videos/challenge.mp4"
    video_output_path = 'video_1_xlt.mp4'

    print(f"正在处理视频: {video_input_path} ...")
    
    # 加载视频并按帧处理
    clip = VideoFileClip(video_input_path)
    out_clip = clip.image_transform(LaneLine_process)
    
    # 输出保存处理后的视频
    out_clip.write_videofile(video_output_path, audio=True)
    print("视频处理完成！")