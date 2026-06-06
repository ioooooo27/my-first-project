from flask import Flask, render_template, request, jsonify, redirect, session
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
import hashlib

app = Flask(__name__)
app.secret_key = "fruit_system_2025"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# 模型
cnn_model = None
yolo_model = YOLO("models/best_yolo_model.pt")

# 账号（无数据库版）
USER = {"user": "123", "admin": "123"}

# 水果科普（API 备用）
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

# ------------------- 页面路由 -------------------
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html', user=session['user'])

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/fresh')
def fresh():
    return render_template('fresh.html')

# ------------------- 登录 -------------------
@app.route('/api/login', methods=['POST'])
def api_login():
    user = request.form['username']
    pwd = request.form['password']
    if user in USER and USER[user] == pwd:
        session['user'] = user
        return jsonify({"code":0})
    return jsonify({"code":1})

# ------------------- 单图识别 -------------------
@app.route('/api/detect', methods=['POST'])
def api_detect():
    file = request.files['file']
    path = f"uploads/{file.filename}"
    file.save(path)
    res = yolo_model(path)
    name = res[0].names[int(res[0].boxes.cls[0])] if len(res[0].boxes) else "unknown"
    info = get_fruit_info(name)
    return jsonify({"name":name, "info":info})

# ------------------- 批量识别 -------------------
@app.route('/api/batch', methods=['POST'])
def api_batch():
    file = request.files['file']
    path = f"uploads/{file.filename}"
    file.save(path)
    res = yolo_model(path)
    count = {}
    for c in res[0].boxes.cls:
        n = res[0].names[int(c)]
        count[n] = count.get(n,0)+1
    return jsonify(count)

# ------------------- 新鲜度识别 -------------------
@app.route('/api/fresh', methods=['POST'])
def api_fresh():
    file = request.files['file']
    path = f"uploads/{file.filename}"
    file.save(path)
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
        level = "不新鲜"
        edible = "不建议食用"
    return jsonify({"score":score, "level":level, "edible":edible})

if __name__ == '__main__':
    app.run(debug=True)