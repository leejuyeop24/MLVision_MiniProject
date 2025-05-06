import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 폴더 경로 설정
banana_folders = [
    '../data/dataset2/banana/',
    '../data/dataset2/unripe/',
    '../data/dataset2/overripe/',
]

non_banana_folders = [
    '../data/dataset2/bananakick/',
    '../data/dataset2/bananakick2/',
    '../data/dataset2/mango/',
    '../data/dataset2/chickoo/',
    '../data/dataset2/grapes/',
]

# 회전 함수
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

# 흑백 특징 추출 (히스토그램 + 통계)
def extract_grayscale_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (120, 120))
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    mean = gray.mean()
    std = gray.std()
    return np.concatenate([hist, [mean, std]])

# 데이터 수집 함수
def collect_data(folder_list, label):
    features = []
    for folder in folder_list:
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, file)
                img = cv2.imread(path)
                if img is None:
                    continue

                # 이미지 전처리/변환
                transforms = [
                    img,
                    cv2.flip(img, 1),
                    rotate_image(img, 15),
                    rotate_image(img, -15),
                    rotate_image(img, 90),
                    cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                ]

                for transformed in transforms:
                    feat = extract_grayscale_features(transformed)
                    row = list(feat) + [label]
                    features.append(row)
    return features

# 바나나/비바나나 데이터 수집
banana_data = collect_data(banana_folders, label=1)
non_banana_data = collect_data(non_banana_folders, label=0)

# 통합
all_data = banana_data + non_banana_data
data = np.array(all_data)
X = data[:, :-1].astype(np.float32)
y = data[:, -1].astype(np.int32)