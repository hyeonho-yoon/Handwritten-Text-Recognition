!apt-get update
!apt-get install -y tesseract-ocr
!apt-get install -y tesseract-ocr-kor
!pip install pytesseract

import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 이미지 불러오기
path = ''
image = cv2.imread(path)


# OCR 실행
data = pytesseract.image_to_data(image, lang='kor', output_type=pytesseract.Output.DATAFRAME)

for _, row in data.iterrows():

    # 인식된 텍스트에 사각형 표시
    if pd.notna(row['text']) and str(row['text']).strip() != '':
        x, y, w, h = row['left'], row['top'], row['width'], row['height']
        # 인식된 텍스트 사각형 표시
        cv2.rectangle(image, (x, y), (x + w, y + h), (1, 1, 1), 5)

# 인식된 텍스트 출력
count = 0
for _, row in data.iterrows():
  count+=1
  #print(f"인식된 텍스트: {row['text']}")

print(f"인식된 텍스트 개수: {count}")


plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV는 BGR이므로 RGB로 변환
plt.axis("off")
plt.show()
