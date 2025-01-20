import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import glob

# 모델과 프로세서 불러오기
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 이미지가 저장된 디렉토리
image_dir = "images/demo"  # 이미지를 분석할 디렉토리 경로

# 하위 폴더 및 파일 검색 (모든 이미지 파일 찾기)
image_files = glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True)
image_files += glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)  # PNG 파일도 포함

# 결과를 파일로 저장하기 위해 열기
with open("output_captions.txt", "w") as output_file:
    # 이미지 파일에 대해 반복하여 캡션 생성
    for idx, image_path in enumerate(image_files):
        try:
            # 이미지 로드
            image = Image.open(image_path)

            # 이미지 캡션 생성
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # 이미지 파일 이름과 캡션을 파일에 기록
            output_file.write(f"Image {idx + 1} ({os.path.basename(image_path)}): {caption}\n")
            print(f"Image {idx + 1}: {caption}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
