!pip install google-cloud-vision

import io
import os
from google.cloud import vision
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Google Cloud 서비스 계정 키 경로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# 클라이언트 생성
client = vision.ImageAnnotatorClient()

# 이미지 경로
image_path = ""

# 저장될 이미지 파일명
output_image_path = ""

# 이미지 파일 읽기 (Google Cloud Vision API)
with io.open(image_path, "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# OCR 요청
response = client.text_detection(image=image)
texts = response.text_annotations

# cv2 이미지 로드
cv2_image = cv2.imread(image_path)


# 결과 출력 및 박스 그리기
if texts:
    print("전체 텍스트:\n", texts[0].description)

    text_counts = 0
    for text in texts[1:]:  # texts[0]은 전체 텍스트 블록
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

        pts = np.array(vertices, np.int32).reshape((-1, 1, 2))

        cv2.polylines(cv2_image, [pts], isClosed=True, color=(1, 1, 1), thickness=3)

        text_counts += 1

    # Matplotlib으로 이미지 표시를 위해 BGR을 RGB로 변환
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    print(f'인식된 텍스트의 총 개수: {text_counts}개')

    # 이미지 저장
    cv2.imwrite(output_image_path, cv2_image)

    # 이미지 표시
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2_image_rgb)
    plt.axis("off")
    plt.show()
else:
    print("텍스트를 찾을 수 없습니다.")
