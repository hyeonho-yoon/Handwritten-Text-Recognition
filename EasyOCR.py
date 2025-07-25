!pip install easyocr

import easyocr
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

# EasyOCR 모델 로드
reader = easyocr.Reader(['ko', 'en'], gpu = True)

# 테스트 이미지 로드
img_path = ''
# 저장될 이미지 파일명
image_save_path = ""
original_img = cv2.imread(img_path)

# OCR 실행
result = reader.readtext(original_img)

img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
font_path = '/content/drive/MyDrive/NanumGothic-Bold.ttf'
font = ImageFont.truetype(font_path, 10)
draw = ImageDraw.Draw(img)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(255, 3), dtype = "uint8")

for i in result :
  x = i[0][0][0]
  y = i[0][0][1]
  w = i[0][1][0] - i[0][0][0]
  h = i[0][2][1] - i[0][1][1]
  text = i[1]

  draw.rectangle(((x, y), (x+w, y+h)), outline=(1, 1, 1), width=4)

img_save = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 이미지 저장
cv2.imwrite(image_save_path, img_save)

# 결과 출력
count = 0
for (bbox, text, prob) in result:
    print(f"인식된 텍스트: {text}")
    count+=1

print(f"인식된 텍스트의 개수: {count}")

plt.figure(figsize=(20, 10))
plt.imshow(img)
plt.show()
