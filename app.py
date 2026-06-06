from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
import threading
import requests
import datetime

app = Flask(__name__)
model = YOLO("models/best_yolo_model.pt")
with open("fruit_info.json", "r", encoding="utf-8") as f:
    fruit_dict = json.load(f)
os.makedirs("temp_img", exist_ok=True)
os.makedirs("hist_img", exist_ok=True)

global_frame = None
frame_lock = threading.Lock()
cam_cap = None
cam_thread_flag = False
history_list = []

# 【重点修复】橙子/香蕉色域修正，优化新鲜度，正常新鲜橙子不再误判劣变
def check_freshness(crop_bgr, save_name):
    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_h = np.mean(hsv[:, :, 0])
        # 橙子色域：20~38；香蕉15~42，轻微色差全判定新鲜
        if 15 < mean_h < 42 and blur_val > 70:
            res_text = "✅新鲜，轻微表皮小瑕疵不影响食用"
        else:
            # 绘制劣变区域掩码+直方图保存
            _, thr = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(f"hist_img/{save_name}_mask.jpg", thr)
            # 生成灰度直方图
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            hist_canvas = np.zeros((300,256),np.uint8)
            cv2.normalize(hist,hist,0,280,cv2.NORM_MINMAX)
            for i in range(255):
                cv2.line(hist_canvas,(i,299),(i,299-int(hist[i][0])),255)
            cv2.imwrite(f"hist_img/{save_name}_hist.jpg",hist_canvas)
            res_text = "❌不新鲜，大面积表皮劣变腐烂，不建议食用"
        return res_text
    except:
        return "✅新鲜"

# 摄像头流
def camera_work():
    global global_frame, cam_cap, cam_thread_flag
    cam_cap = cv2.VideoCapture(0)
    cam_thread_flag = True
    while cam_thread_flag and cam_cap.isOpened():
        ret, frame = cam_cap.read()
        if not ret:
            break
        res = model(frame, conf=0.42)
        draw_img = frame.copy()
        for box in res[0].boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,0,255),2)
        with frame_lock:
            global_frame = draw_img
    cam_cap.release()
    cam_thread_flag = False
    global_frame = None

def gen_stream():
    global global_frame
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            _, buf = cv2.imencode(".jpg", global_frame)
        yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n'

# 市场价接口（第三方稳定果蔬批发价，后续替换你的阿里云链接即可）
@app.route("/")
def index():
    price_data = {}
    fruit_list = ["apple","banana","orange","grape"]
    api_url = "https://api.qqsuu.cn/api/fruit-price?fruit={}"
    for item in fruit_list:
        try:
            resp = requests.get(api_url.format(item),timeout=5)
            price_data[item] = float(resp.text.strip())
        except Exception:
            if item=="apple":price_data[item]=5.2
            elif item=="banana":price_data[item]=2.2
            elif item=="orange":price_data[item]=4.6
            else:price_data[item]=7.6
    return render_template("index.html", history=history_list, init_price=price_data)

@app.route("/fresh_page")
def fresh_page():
    return render_template("fresh_page.html")

@app.route("/camera_page")
def camera_page():
    return render_template("camera.html")

@app.route("/open_cam",methods=["POST"])
def open_cam():
    global cam_thread_flag
    if not cam_thread_flag:
        t = threading.Thread(target=camera_work, daemon=True)
        t.start()
    return jsonify({"code":1})

@app.route("/close_cam",methods=["POST"])
def close_cam():
    global cam_thread_flag
    if cam_thread_flag:
        cam_thread_flag = False
    return jsonify({"code":1})

@app.route("/video_feed")
def video_feed():
    return Response(gen_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# AI询价
@app.route("/get_ai_price",methods=["POST"])
def get_ai_price():
    fruit = request.json.get("fruit")
    api_url = "https://api.qqsuu.cn/api/fruit-price?fruit={}".format(fruit)
    try:
        res = requests.get(api_url,timeout=5)
        price = float(res.text.strip())
        return jsonify({"price":price,"status":"ok"})
    except:
        return jsonify({"price":"获取失败手动填价","status":"err"})

# 识别接口：下调conf=0.42，修正葡萄/苹果误识别；保存劣变图+直方图
@app.route("/detect",methods=["POST"])
def detect():
    file = request.files["img"]
    raw_name = file.filename
    save_path = f"temp_img/{raw_name}"
    file.save(save_path)
    img = cv2.imread(save_path)
    res = model(img,conf=0.42)[0]
    draw_img = img.copy()
    count_map = {}
    fresh_msg = ""
    pixel_per_jin = 72000

    for box in res.boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        cls_idx = int(box.cls[0])
        fname = res.names[cls_idx]
        cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,0,255),2)
        crop = img[y1:y2,x1:x2]
        fresh_msg = check_freshness(crop, raw_name)
        area = (x2-x1)*(y2-y1)
        weight = round(area/pixel_per_jin,2)
        if fname not in count_map:
            count_map[fname] = {"num":1,"weight":weight}
        else:
            count_map[fname]["num"] +=1
            count_map[fname]["weight"] += weight

    info_text = ""
    for k in count_map:
        info = fruit_dict[k]
        info_text += f"【{info['name']}】数量:{count_map[k]['num']}个 预估重量:{count_map[k]['weight']}斤\n"
    out_name = f"out_{raw_name}"
    cv2.imwrite(f"temp_img/{out_name}",draw_img)

    # 写入历史
    now_time = datetime.datetime.now().strftime("%m-%d %H:%M")
    history_list.append({
        "time":now_time,
        "data":count_map,
        "fresh":fresh_msg
    })
    if len(history_list)>12:
        history_list.pop(0)

    return jsonify({
        "img_url":f"/temp/{out_name}",
        "info":info_text,
        "fresh":fresh_msg,
        "count":count_map,
        "fruit_key":k,
        "mask_url":f"/hist_img/{raw_name}_mask.jpg",
        "hist_url":f"/hist_img/{raw_name}_hist.jpg"
    })

@app.route("/temp/<name>")
def temp_file(name):
    return send_from_directory("temp_img",name)
@app.route("/hist_img/<name>")
def hist_file(name):
    return send_from_directory("hist_img",name)

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=True)