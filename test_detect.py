from ultralytics import YOLO
import cv2
import json
import numpy as np

# 1. 加载训练完的最优模型
model = YOLO("app/models/best_yolo_model.pt")

# 2. 加载科普文档
with open("app/fruit_info.json", "r", encoding="utf-8") as f:
    fruit_info = json.load(f)

# 3. 测试图片（根目录test.jpg）
img_path = "test.jpg"
results = model(img_path, conf=0.5)

# 读取原图，手动统一画【红色框BGR:(0,0,255)】
img_raw = cv2.imread(img_path)

for res in results:
    detect_result = {}
    # 遍历所有目标框
    for box in res.boxes:
        # 获取框坐标、类别
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_index = int(box.cls[0])
        fruit_name = res.names[cls_index]
        # 绘制红框
        cv2.rectangle(img_raw, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # 统计数量
        detect_result[fruit_name] = detect_result.get(fruit_name, 0) + 1

    # 弹窗展示红框图片
    cv2.imshow("水果识别（红框标注）", img_raw)

    print("📊 批量识别统计：", detect_result)
    print("\n=====水果科普详情=====")
    for fruit_key in detect_result:
        data = fruit_info[fruit_key]
        print(f"【{data['name']}】")
        print(f"数量：{detect_result[fruit_key]}个")
        print(f"产地：{data['origin']}")
        print(f"营养：{data['nutrition']}")
        print(f"应季：{data['season']} | {data['is_season']}")
        print(f"吃法：{data['eat_way']}")
        print(f"储存方案：{data['save_way']}\n")

cv2.waitKey(0)
cv2.destroyAllWindows()