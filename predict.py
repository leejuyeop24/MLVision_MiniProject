#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cgi
import cgitb
import cv2
import numpy as np
import joblib
from PIL import Image, ImageFile
from io import BytesIO
import sys
sys.stdout.reconfigure(encoding='utf-8')

# CGI 디버그
cgitb.enable()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# HTTP 응답 헤더
print("Content-Type: text/html\n")
print("<html><head><meta charset='utf-8'><title>예측 결과</title></head><body>")

# 이미지 수신
form = cgi.FieldStorage()
if 'image' not in form:
    print("<h2 style='color:red;'>이미지를 업로드하지 않았습니다.</h2></body></html>")
    exit()

file_item = form['image']
image_bytes = file_item.file.read()

try:
    # 이미지 열기
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    # ✅ 학습과 동일한 특징 추출 함수 사용
    def extract_color_features(img):
        img = cv2.resize(img, (120, 120))  # 학습 시에도 이 크기 사용했음
        chans = cv2.split(img)
        features = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            mean = chan.mean()
            std = chan.std()
            features.extend(hist)
            features.extend([mean, std])
        return np.array(features).reshape(1, -1)

    features = extract_color_features(image_np)

    # 모델 로드
    model = joblib.load("xgb_model.pkl")
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    label_map = {0: '기타 과일', 1: '바나나'}
    print(f"<h2>🟢 예측 결과: <span style='color:green'>{label_map[pred]}</span></h2>")
    
except Exception as e:
    print(f"<h2 style='color:red;'>❗ 예측 중 오류 발생:<br>{e}</h2>")

print("</body></html>")

