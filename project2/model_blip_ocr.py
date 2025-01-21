from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import easyocr
import numpy as np

# 모델과 프로세서 불러오기
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# EasyOCR 리더 생성
reader = easyocr.Reader(['en', 'ko'])  # 영어와 한국어 지원

# 이미지 로드
image = Image.open("images/demo/demo7.jpg")

# PIL 이미지를 numpy 배열로 변환
image_np = np.array(image)

# EasyOCR을 사용하여 이미지에서 텍스트 추출
ocr_result = reader.readtext(image_np)

# OCR로 추출된 텍스트 합치기
extracted_text = ' '.join([detection[1] for detection in ocr_result])
print(f"Extracted Text from OCR: {extracted_text}")

# 이미지 캡션 생성
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

# OCR과 함께 캡션 출력
print(f"Caption Generated by BLIP: {caption}")
