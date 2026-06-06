import os
from pathlib import Path

# -------------------------- 配置（直接用，不用改） --------------------------
ROOT = Path("fruit_images_split")  # 你的数据集根目录
CLASSES = ["apple", "orange", "grape", "banana"]  # 加上香蕉
# --------------------------------------------------------------------------------

def create_labels_for_split(split_name):
    img_dir = ROOT / split_name / "images"
    label_dir = ROOT / split_name / "labels"
    label_dir.mkdir(exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png")
    img_paths = []
    for ext in exts:
        img_paths.extend(img_dir.glob(ext))

    print(f"\n[{split_name}] 找到 {len(img_paths)} 张图片，开始生成标签...")

    for img_path in img_paths:
        name_lower = img_path.stem.lower()

        # 识别所有水果关键词，包括香蕉
        cls_id = -1
        if "apple" in name_lower or "苹果" in name_lower:
            cls_id = 0
        elif "orange" in name_lower or "橙子" in name_lower:
            cls_id = 1
        elif "grape" in name_lower or "葡萄" in name_lower:
            cls_id = 2
        elif "banana" in name_lower or "香蕉" in name_lower:  # 加上香蕉
            cls_id = 3

        if cls_id == -1:
            print(f"⚠️ 无法识别类别，跳过：{img_path.name}")
            continue

        # YOLO格式：class_id x_center y_center width height（归一化）
        # 这里假设整图都是目标 → 框是整张图
        txt_content = f"{cls_id} 0.5 0.5 1.0 1.0\n"
        txt_path = label_dir / f"{img_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

    print(f"✅ [{split_name}] 标签生成完成，保存到 {label_dir}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        create_labels_for_split(split)
    print("\n🎉 全部标签生成完毕！")