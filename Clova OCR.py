import requests
import uuid
import time
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# OCR API 엔드포인트
api_url = ''
secret_key = ''

image_file = ''

request_json = {
    'images': [
        {
            'format': 'jpg',
            'name': 'demo'
        }
    ],
    'requestId': str(uuid.uuid4()),
    'version': 'V2',
    'timestamp': int(round(time.time() * 1000))
}

payload = {'message': json.dumps(request_json).encode('UTF-8')}
files = [
  ('file', open(image_file,'rb'))
]
headers = {
  'X-OCR-SECRET': secret_key
}

# OCR API로 POST 요청 전송
response = requests.request("POST", api_url, headers=headers, data = payload, files = files)
result = response.json()

# 원본 이미지 로드
image = cv2.imread(image_file)

text_count = 0
# 인식된 텍스트 박스 그리기
for field in result['images'][0]['fields']:
    text = field['inferText']
    vertices = field['boundingPoly']['vertices']

    pts = [(v['x'], v['y']) for v in vertices]
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))

    # 박스와 텍스트 표시
    cv2.polylines(image, [pts], isClosed=True, color=(1, 1, 1), thickness=3)

    text_count += 1 # 텍스트 개수 세기

    print(text)


print(f'인식된 텍스트의 총 개수: {text_count}개')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 12))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
