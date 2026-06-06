from ultralytics import YOLO
import os
import shutil

# 用绝对路径，彻底避免相对路径坑
data_yaml_path = r"D:\123\水果预测\fruit_images_split\data.yaml"

model = YOLO("yolov8n.pt")

results = model.train(
    data=data_yaml_path,
    epochs=5,
    imgsz=640,
    batch=4,
    device="cpu",
    project=r"D:\123\runs\detect",
    name="train"
)

src = r"D:\123\runs\detect\train\weights\best.pt"
dst = r"D:\123\水果预测\models\best_yolo_model.pt"
os.makedirs(os.path.dirname(dst), exist_ok=True)
if os.path.exists(src):
    shutil.copy(src, dst)
    print("✅ 模型已保存到 models/best_yolo_model.pt")