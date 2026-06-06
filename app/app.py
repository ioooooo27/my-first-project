from flask import Flask, render_template, request, jsonify, send_from_directory, Response, redirect, session
from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
import threading
import requests
import datetime
import sqlite3
import hashlib
import secrets
import time
import base64


app = Flask(__name__)
app.secret_key = "fruit_system_2025"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
model = YOLO(os.path.join(BASE_DIR, "models/best_yolo_model.pt"))
with open(os.path.join(BASE_DIR, "fruit_info.json"), "r", encoding="utf-8") as f:
    fruit_dict = json.load(f)
temp_img_dir = os.path.join(BASE_DIR, "temp_img")
hist_img_dir = os.path.join(BASE_DIR, "hist_img")
os.makedirs(temp_img_dir, exist_ok=True)
os.makedirs(hist_img_dir, exist_ok=True)

# TODO(security): SQLite Database setup for storing credentials securely using scrypt hashing
def get_password_hash(password, salt_hex):
    salt = bytes.fromhex(salt_hex)
    return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=16384, r=8, p=1).hex()

# 调用云端 VLM API 识别新鲜度
def get_freshness_from_cloud(image_path):
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    # 支持从配置文件 config.json 中读取 API Key
    config_path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                gemini_key = gemini_key or cfg.get("GEMINI_API_KEY")
                openai_key = openai_key or cfg.get("OPENAI_API_KEY")
        except Exception:
            pass

    if not gemini_key and not openai_key:
        return None

    try:
        with open(image_path, "rb") as image_file:
            img_data = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = (
            "你是一个水果质检专家。分析图中水果的新鲜度（如有无腐烂、变质、严重破损或大面积黑斑）。"
            "请严格返回一个 JSON 对象，不得包含任何 markdown 格式或额外文本解释，格式必须如下：\n"
            "{\"score\": 85, \"level\": \"新鲜\", \"edible\": \"可食用\", \"reason\": \"说明判定原因（中文）\"}\n"
            "注：score为0-100整数；level只能为'新鲜'、'一般'或'不新鲜'之一；edible只能为'可食用'或'不建议食用'之一。"
        )

        # 优先使用 OpenAI GPT-4o-mini
        if openai_key:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_data}"
                                }
                            }
                        ]
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 150
            }
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                data = json.loads(content.strip())
                return {
                    "score": int(data.get("score", 70)),
                    "level": str(data.get("level", "新鲜")),
                    "edible": str(data.get("edible", "可食用")),
                    "reason": str(data.get("reason", "通过 OpenAI 分析"))
                }

        # 其次使用 Gemini 1.5 Flash
        if gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": img_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                content = res.json()['candidates'][0]['content']['parts'][0]['text']
                data = json.loads(content.strip())
                return {
                    "score": int(data.get("score", 70)),
                    "level": str(data.get("level", "新鲜")),
                    "edible": str(data.get("edible", "可食用")),
                    "reason": str(data.get("reason", "通过 Gemini 分析"))
                }
    except Exception as e:
        print(f"调用云端大模型 API 失败: {e}")
    return None

def init_db():
    db_path = os.path.join(BASE_DIR, "users.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    """)
    conn.commit()
    
    # Check if we need to seed
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        # Seed default users: user/123, admin/123
        default_users = [("user", "123"), ("admin", "123")]
        for username, password in default_users:
            salt_hex = secrets.token_hex(16)
            pwd_hash = get_password_hash(password, salt_hex)
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, pwd_hash, salt_hex)
            )
        conn.commit()
    conn.close()

# Initialize local database
init_db()

def get_fruit_info(name):
    info = {
        "apple": {"营养":"富含维生素C","产地":"山东、新疆","应季":"秋季","吃法":"直接吃","存储":"冷藏"},
        "banana":{"营养":"富含钾","产地":"海南、广东","应季":"全年","吃法":"直接吃","存储":"阴凉处"},
        "orange":{"营养":"维C丰富","产地":"江西、湖南","应季":"冬季","吃法":"直接吃","存储":"冷藏"},
        "grape":{"营养":"抗氧化","产地":"新疆、山东","应季":"夏季","吃法":"直接吃","存储":"冷藏"},
        "strawberry":{"营养":"维生素丰富","产地":"四川、辽宁","应季":"春季","吃法":"直接吃","存储":"冷藏"},
        "pear":{"营养":"润肺止咳","产地":"河北、山东","应季":"秋季","吃法":"直接吃","存储":"冷藏"}
    }
    return info.get(name, info["apple"])

global_frame = None
frame_lock = threading.Lock()
cam_cap = None
cam_thread_flag = False
history_list = []

# 摄像头实时结果共享与连接数管理
global_cam_results = {}
cam_results_lock = threading.Lock()
active_connections = 0
active_connections_lock = threading.Lock()

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
            cv2.imwrite(os.path.join(hist_img_dir, f"{save_name}_mask.jpg"), thr)
            # 生成灰度直方图
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            hist_canvas = np.zeros((300,256),np.uint8)
            cv2.normalize(hist,hist,0,280,cv2.NORM_MINMAX)
            for i in range(255):
                cv2.line(hist_canvas,(i,299),(i,299-int(hist[i][0])),255)
            cv2.imwrite(os.path.join(hist_img_dir, f"{save_name}_hist.jpg"), hist_canvas)
            res_text = "❌不新鲜，大面积表皮劣变腐烂，不建议食用"
        return res_text
    except:
        return "✅新鲜"

# 摄像头流
def camera_work():
    global global_frame, cam_cap, cam_thread_flag, global_cam_results
    cam_cap = cv2.VideoCapture(0)
    
    # 帧过滤机制：跳帧以极大降低CPU开销，保持流画面流畅，无累积延迟
    frame_count = 0
    latest_res = None
    
    while cam_thread_flag and cam_cap.isOpened():
        ret, frame = cam_cap.read()
        if not ret:
            break
            
        frame_count += 1
        # 每隔 3 帧做一次 YOLO 推理 (约 10 FPS)，其余帧复用上一帧的检测框
        if latest_res is None or frame_count % 3 == 0:
            latest_res = model(frame, conf=0.42)[0]
            
        draw_img = frame.copy()
        current_counts = {}
        
        if latest_res is not None:
            for box in latest_res.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cls_idx = int(box.cls[0])
                fname = latest_res.names[cls_idx]
                
                # 统计数量
                chinese_name = fruit_dict.get(fname, {}).get("name", fname)
                current_counts[chinese_name] = current_counts.get(chinese_name, 0) + 1
                
                # 画红色矩形框
                cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,0,255),2)
                # 在图像上绘制英文类别和置信度标签
                conf_val = float(box.conf[0])
                label = f"{fname.capitalize()} {conf_val:.2f}"
                cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        with frame_lock:
            global_frame = draw_img
        with cam_results_lock:
            global_cam_results = current_counts

    if cam_cap:
        cam_cap.release()
    cam_thread_flag = False
    with frame_lock:
        global_frame = None
    with cam_results_lock:
        global_cam_results = {}

def gen_stream():
    global global_frame, cam_thread_flag, active_connections
    with active_connections_lock:
        active_connections += 1
    try:
        while True:
            time.sleep(0.03)  # 控制帧率(约30fps)并防止CPU 100%空转及锁饥饿
            if not cam_thread_flag:
                break
            with frame_lock:
                if global_frame is None:
                    continue
                _, buf = cv2.imencode(".jpg", global_frame)
            yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n'
    except GeneratorExit:
        pass
    finally:
        with active_connections_lock:
            active_connections -= 1
            if active_connections <= 0:
                cam_thread_flag = False

# 获取新发地水果市场价格数据的辅助函数
def get_realtime_fruit_price(fruit_name):
    if not fruit_name:
        return None
    fruit_name = str(fruit_name).strip().lower()
    
    # 映射表，支持英文、中文拼音、汉字及常见品名
    mapping = {
        "apple": "富士",
        "苹果": "富士",
        "富士": "富士",
        "banana": "香蕉",
        "香蕉": "香蕉",
        "orange": "橙子",
        "橙子": "橙子",
        "grape": "葡萄",
        "葡萄": "葡萄"
    }
    
    # 1. 优先精确匹配
    query_name = mapping.get(fruit_name)
    
    # 2. 如果没有精确匹配，尝试模糊匹配（例如输入“红富士苹果”或“大香蕉”）
    if not query_name:
        for key, val in mapping.items():
            if key in fruit_name or fruit_name in key:
                query_name = val
                break
                
    if not query_name:
        return None
    url = "http://www.xinfadi.com.cn/getPriceData.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    data = {
        "limit": 1,
        "current": 1,
        "prodName": query_name
    }
    try:
        res = requests.post(url, data=data, headers=headers, timeout=5)
        if res.status_code == 200:
            result_json = res.json()
            items = result_json.get("list", [])
            if items:
                return float(items[0]["avgPrice"])
    except Exception as e:
        print(f"获取新发地 [{query_name}] 价格失败: {e}")
    return None

# 市场价接口（北京新发地批发价格，并提供离线降级方案）
@app.route("/")
def login():
    if 'user' in session:
        return redirect('/index')
    return render_template('login.html')

@app.route("/index")
def index():
    if 'user' not in session:
        return redirect('/')
    price_data = {}
    fruit_list = ["apple","banana","orange","grape"]
    for item in fruit_list:
        price = get_realtime_fruit_price(item)
        if price is not None:
            price_data[item] = price
        else:
            if item == "apple": price_data[item] = 5.2
            elif item == "banana": price_data[item] = 2.2
            elif item == "orange": price_data[item] = 4.6
            else: price_data[item] = 7.6
    return render_template("index.html", history=history_list, init_price=price_data, user=session['user'])

@app.route("/fresh_page")
def fresh_page():
    if 'user' not in session:
        return redirect('/')
    return render_template("fresh_page.html")

@app.route("/camera_page")
def camera_page():
    if 'user' not in session:
        return redirect('/')
    return render_template("camera.html")

@app.route("/open_cam",methods=["POST"])
def open_cam():
    global cam_thread_flag
    if not cam_thread_flag:
        cam_thread_flag = True
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

@app.route("/api/cam_results")
def api_cam_results():
    with cam_results_lock:
        return jsonify(global_cam_results)

# AI询价 (使用新发地实时均价，获取失败时返回友好降级提示)
@app.route("/get_ai_price",methods=["POST"])
def get_ai_price():
    fruit = request.json.get("fruit")
    price = get_realtime_fruit_price(fruit)
    if price is not None:
        return jsonify({"price": price, "status": "ok"})
    else:
        return jsonify({"price": "获取失败手动填价", "status": "err"})

# 识别接口：下调conf=0.42，修正葡萄/苹果误识别；保存劣变图+直方图
@app.route("/detect",methods=["POST"])
def detect():
    file = request.files["img"]
    raw_name = file.filename
    save_path = os.path.join(temp_img_dir, raw_name)
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
    cv2.imwrite(os.path.join(temp_img_dir, out_name), draw_img)

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
    return send_from_directory(temp_img_dir, name)
@app.route("/hist_img/<name>")
def hist_file(name):
    return send_from_directory(hist_img_dir, name)

# ------------------- 登录 API -------------------
@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        user = request.form['username']
        pwd = request.form['password']
        
        db_path = os.path.join(BASE_DIR, "users.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Parameterized query to completely prevent SQL injection
        cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (user,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            stored_hash, salt_hex = row
            pwd_hash = get_password_hash(pwd, salt_hex)
            if pwd_hash == stored_hash:
                session['user'] = user
                return jsonify({"code":0})
        return jsonify({"code":1})
    except Exception:
        # TODO(security): Expose only generic error message, log diagnostic internally if needed
        return jsonify({"code":1})

# ------------------- 单图、批量、新鲜度简单页面及 API -------------------
@app.route('/detect', methods=['GET'])
def detect_page():
    if 'user' not in session:
        return redirect('/')
    return render_template('detect.html')

@app.route('/batch')
def batch_page():
    if 'user' not in session:
        return redirect('/')
    return render_template('batch_detect.html')

@app.route('/fresh')
def fresh():
    if 'user' not in session:
        return redirect('/')
    return render_template('fresh.html')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    res = model(path)
    name = res[0].names[int(res[0].boxes.cls[0])] if len(res[0].boxes) else "unknown"
    info = get_fruit_info(name)
    return jsonify({"name":name, "info":info})

@app.route('/api/batch', methods=['POST'])
def api_batch():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    res = model(path)
    count = {}
    for c in res[0].boxes.cls:
        n = res[0].names[int(c)]
        count[n] = count.get(n,0)+1
    return jsonify(count)

@app.route('/api/fresh', methods=['POST'])
def api_fresh():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    
    # 优先使用云端大模型 VLM API 进行分析
    cloud_result = get_freshness_from_cloud(path)
    if cloud_result:
        return jsonify({
            "score": cloud_result["score"],
            "level": cloud_result["level"],
            "edible": cloud_result["edible"],
            "reason": cloud_result["reason"],
            "source": "cloud"
        })
        
    # 如果没有配置云端 API 或调用失败，降级使用本地离线机制
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = int(np.mean(gray))
    if score > 100: score=100
    if score >= 70:
        level = "新鲜"
        edible = "可食用"
    elif score >= 40:
        level = "一般"
        edible = "可食用"
    else:
        level = "not_fresh"
        level = "不新鲜"
        edible = "不建议食用"
    return jsonify({
        "score": score,
        "level": level,
        "edible": edible,
        "reason": "本地离线简易检测（配置 config.json 内的 API Key 可激活云端大模型质检）",
        "source": "local"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5002,debug=True)