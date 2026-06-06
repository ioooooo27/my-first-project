import os
import requests
import random
from bs4 import BeautifulSoup
import shutil

# 配置
fruits = ["apple", "banana", "orange", "grape", "strawberry", "pear"]
root = "fruit_images"
split_root = "fruit_images_split"
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

os.makedirs(root, exist_ok=True)
for f in fruits:
    os.makedirs(os.path.join(root, f), exist_ok=True)

# 简单爬虫（百度图片，示例，少量图）
def crawl_fruit(fruit_name):
    url = f"https://image.baidu.com/search/index?tn=baiduimage&word={fruit_name}"
    headers = {"User-Agent":"Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    imgs = soup.find_all("img")
    save_dir = os.path.join(root, fruit_name)
    cnt = 0
    for img in imgs:
        try:
            src = img.get("src")
            if not src or not src.startswith("http"):
                continue
            img_data = requests.get(src, timeout=5).content
            with open(os.path.join(save_dir, f"{cnt}.jpg"), "wb") as f:
                f.write(img_data)
            cnt += 1
            if cnt >= 50:
                break
        except:
            continue
    print(f"{fruit_name} 爬了 {cnt} 张")

for fruit in fruits:
    crawl_fruit(fruit)

# 划分数据集
def split_data():
    random.seed(0)
    for fruit in fruits:
        src_dir = os.path.join(root, fruit)
        imgs = [f for f in os.listdir(src_dir) if f.endswith(("jpg","png"))]
        random.shuffle(imgs)
        total = len(imgs)
        train_num = int(total*train_ratio)
        val_num = int(total*val_ratio)

        for phase in ["train","val","test"]:
            os.makedirs(os.path.join(split_root, phase, fruit), exist_ok=True)

        for i, img in enumerate(imgs):
            src = os.path.join(src_dir, img)
            if i < train_num:
                dst = os.path.join(split_root, "train", fruit, img)
            elif i < train_num+val_num:
                dst = os.path.join(split_root, "val", fruit, img)
            else:
                dst = os.path.join(split_root, "test", fruit, img)
            shutil.copy(src, dst)
    print("数据集划分完成")

split_data()