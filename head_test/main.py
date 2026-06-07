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
# 银白的月光洒在地上，到处都有蟋蟀的凄切的叫声。夜的香气弥漫在空中，织成了一个柔软的网，把所有的景物都罩在里面。眼睛所接触到的都是罩上这个柔软的网的东西，任是一草一木，都不是象在白天里那样地现实了，它们都有着模糊、空幻的色彩，每一样都隐藏了它的细致之点，都保守着它的秘密，使人有一种如梦如幻的感觉。
#
# （巴金《家》）
#
# 月光如银子，无处不可照及，山上竹篁在月光下变成了一片黑色。身边草丛中虫声繁密如落雨。间或不知道从什么地方，忽然会有一只草莺“落落落落嘘”啭着它的喉咙，不久之间，这小鸟儿又好象明白这是半夜，不应当那么吵闹，便仍然闭着那小小眼儿安睡了。
#
# （沈从文《边城》）
#
# 他靠纱窗望出去。满天的星又密又忙，它们声息全无，而看来只觉得天上热闹。一梳月亮象形容未长成的女孩子，但见人已不羞缩，光明和轮廓都清新刻露，渐渐可烘衬夜景。小园草地里的小虫琐琐屑屑地在夜谈。不知哪里的蛙群齐心协力地干号，象声浪给火煮得发沸。几星萤火优游来去，不象飞行，象在厚密的空气里漂浮，月光不到的阴黑处，一点萤火忽明，象夏夜的一只微绿的小眼睛。
#
# （钱钟书《围城》）
#
# 中山公园的水池象是一面镜子，圆圆的月亮映在池面。池子附近树旁的几盏路灯，那圆圆的灯光映在水里，就象是一个小月亮似的，围绕着池中的月亮。一片一片臃肿的白云缓缓地移过池面，仿佛是一群老妇，弯着背，一步一步吃力地从月亮前面走过，想把月亮遮住，月亮却透过云片的空隙倾泻下皎洁的光芒。一片白云和一片白云连起，如同一条宽大的不规则的带子，给澄澄的天空分成两半。白云移过，逐渐消逝在远方。天空碧澄澄的，月亮显得分外皎洁。
#
# （周而复《上海的早晨》）
#
# 五月末的北方夜晚，是最清新、最美好的时刻。天空象是刷洗过一般，没有一丝云雾，蓝晶晶的，又高又远。一轮圆圆的月亮，从东边的山梁上爬出来，如同一盏大灯笼，把个奇石密布的山谷照得亮堂堂，把树枝、幼草的影投射在小路上，花花点点，悠悠荡荡。宿鸟在枝头上叫着，小虫子在草棵子里蹦着，梯田里春苗在拔秆儿生长着；山野中也有万千生命在欢腾着……
#
# （浩然《艳阳天》）
#
# 月光洒满了这园庭，远处的树林，顶上载着银色的光华，林里烘出浓厚的黑影，寂静严肃的压在那里。喷水池的喷水，池里的微波，都反射着皎洁的月光，在那里荡漾，她脚下的绿茵和近旁的花草也披了月光，柔软无声的在受她的践踏。
#
# （郁达夫《秋河》）
#
# 月亮快要出来了。月亮还远着呢，可是在地平线后边，人们觉得它从黑暗的深渊上升。一道微弱的光，给围绕在高坡上的树顶镶了一条花边，好象高脚杯的边缘，这些反映在微光中的树峰的侧影，一分钟比一分钟显得更为深黑。
#
# （法 罗曼•罗兰：《母与子》）
#
# 雾霭消散了，银色的月光好象一身自得耀眼的寡妇的丧服，覆盖着广阔的沙滩。河面没有一条船只，甚至看不见一丝微波，河心河岸，到处是一片宁静，这宁静有如死亡带给受尽苦难的病患者的一种无休止的安宁。
#
# （印度 泰戈尔：《沉船》）
#
# 过了八公里的瞿塘峡，乌沉沉的云雾，突然隐去，峡顶上一道蓝天，浮着几小片金色浮云，一注阳光像闪电样落在左边峭壁上。右面峰顶上一片白云像白银片样发亮了，但阳光还没有降临。这时，远远前方，无数层峦叠嶂之上，迷蒙云雾之中，忽然出现一团红雾。你看，绛紫色的山峰，衬托着这一团雾，真美极了。就像那深谷之中向上反射出红色宝石的闪光，令人仿佛进入了神话境界。这时，你朝江流上望去，也是色彩缤纷：两面巨岩，倒影如墨；中间曲曲折折，却像有一条闪光的道路，上面荡着细碎的波光；近处山峦，则碧绿如翡翠。时间一分钟一分钟过去，前面那团红雾更红更亮了。船越驶越近，渐渐看清有一高峰亭亭笔立于红雾之中，渐渐看清那红雾原来是千万道强烈的阳光。八点二十分，我们来到这一片晴朗的金黄色朝阳之中。
#
# 刘白羽《长江三日》
#
# 隔断了众人与我的是漫天的雾。任是高屋崇楼，如水的车辆，拥挤的行人；一切都不复存在，连自己行走时摇荡出去的手臂也消失在迷茫之中了。
#
# 靳以《雾》
#
# 屋子外面，原是浓厚得对面不见人影的晨雾，这时已经消退，变淡了。慢慢得势的阳光里，白蒙蒙的雾点子，一阵一阵地翻腾，飘散，好像沙沙有声。篱笆，土堆，墙头，都在雾气里显出模糊的形象。
#
# 王西彦《春回地暖》
#
# 雾霭
#
# 像轻纱，像烟岚，像云彩；挂在树上，绕在屋脊，漫在山路上，藏在草丛中。一会儿像奔涌的海潮，一会儿像白鸥在翻飞。霞烟阵阵，浮去飘来，一切的一切，变得朦朦胧胧的了。顷刻间，这乳白色的轻霭，化成小小的水滴。洒在路面上，洒在树丛中，洒在人头脸上。轻轻的，腻腻的，有点潮湿。人们吸进这带有野菊花药香味儿的气息，觉得有点微醺。
#
# 仇智杰《雾纱赋》
#
# 晨雾
#
# 夜雾慢慢淡了，颜色变白，像是流动着的透明体，东方发白了。浮动着的轻纱一般的迷雾笼罩着曹阳新村，新村的建筑和树木若有若无。说它有吧，看不到那些建筑和树木的整体；说它没有吧，迷雾开豁的地方，又隐隐露出建筑和树木部分的轮廓，随着迷雾的浓淡，变幻多姿，仿佛是海市蜃楼。
#
# 周而复《上海的早晨》
#
# 不知什么时候起了雾。黎明时分，浓雾像棉团似的从上游滚滚而来；爬上河岸，越上树丛，向两侧泛滥开去……浓雾塞满了小棚，沾在脸上湿漉漉的、滑腻腻的；我们谁也看不清谁的脸。
#
# 叶蔚林《在没有航标的河流上》
#
# 有一个浓雾的早晨，我来到堤边。四处迷迷茫茫，山和湖都不见了，面前只有看不透的乳白色的混沌。唉乃之声由远而近，和悦耳的鸟声相应和。白色的空洞里隐隐约约有一个点子，而后，一只船的轮廓渐渐显露出来。这是这一天最早的一只游艇。
#
# 于敏《西湖即景》
#
# 清晨，浓雾弥漫。依照医生的嘱咐，我在湖滨悠闲地散步。耳边只闻鸟鸣，百啭千声，都看不见它们玲珑身影。一团团微带寒意的浓雾不时扑在脸上，掠过身旁。平日那装着耀眼的高压水银灯泡的路灯，今天显得那么暗淡无力，在翻腾缭绕的雾气中闪烁迷离。我仿佛正走进一个童话世界。
#
# 张平《镜湖晨雾》
#
# 夜雾
#
# 有一回从滑雪会走回松雪楼，忽然察觉路上有一层雾，一下子浓了过来，一下子又散了开去，那真是一种奇妙的经验，仿佛走进一个雾帐，雾自发边流过，自耳际流过，自指间流过，都感觉得到；又仿佛行舟在一条雾河，两旁的松涛声鸣不住，轻舟一转，已过了万重山，回首再望，已看不见有雾来过，看不见雾曾在此驻留了。
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