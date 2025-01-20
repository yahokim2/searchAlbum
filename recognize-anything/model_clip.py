import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# 분석할 이미지 로드
image = Image.open("images/demo/demo1.jpg")  # 분석할 이미지 경로로 수정

# 텍스트와 이미지를 입력으로 준비
texts = ["a photo of a lake", "a photo of a man", "a photo of a train"]  # 다양한 텍스트 설명 예시

# 이미지와 텍스트를 모델에 입력할 수 있는 형식으로 전처리
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 모델 실행
with torch.no_grad():
    outputs = model(**inputs)

# 이미지와 텍스트의 유사도를 계산 (이미지와 각 텍스트 간의 유사도)
logits_per_image = outputs.logits_per_image  # 이미지와 텍스트 간의 유사도 점수
probs = logits_per_image.softmax(dim=1)  # 확률로 변환

# 가장 높은 유사도를 가진 텍스트 출력
similarity_scores = probs[0].tolist()  # 확률을 리스트로 변환
best_match_idx = similarity_scores.index(max(similarity_scores))
print(f"The best matching text for the image is: '{texts[best_match_idx]}'")
