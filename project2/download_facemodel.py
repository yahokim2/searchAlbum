
# Hugging Face에서 vit-base-patch16-224-in21k 모델을 다운로드하는 방법을 안내드립니다. 이 모델은 Vision Transformer (ViT) 모델 중 하나로 이미지 분류 작업에 사용됩니다. Hugging Face의 transformers 라이브러리를 사용하여 이 모델을 쉽게 다운로드하고 사용할 수 있습니다.

# 1. Hugging Face 모델 다운로드 준비
# 먼저, Hugging Face의 모델을 다운로드하려면 transformers와 torch 라이브러리를 설치해야 합니다. 만약 설치하지 않았다면, 아래 명령어로 설치할 수 있습니다.

# 1.1. 필요한 라이브러리 설치
# 터미널에서 다음 명령어를 실행하여 필요한 라이브러리를 설치합니다:

# bash
# 코드 복사
# pip install transformers torch
# 2. ViT 모델 다운로드 및 사용
# vit-base-patch16-224-in21k 모델을 다운로드하려면, transformers 라이브러리를 사용하여 모델을 불러옵니다. 아래는 이 모델을 다운로드하고 사용하는 기본 코드입니다.

# 2.1. 모델 및 토크나이저 불러오기
# python
# 코드 복사
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# ViT 모델과 Feature Extractor 불러오기
model_name = "google/vit-base-patch16-224-in21k"

# 모델과 Feature Extractor 다운로드
model = AutoModelForImageClassification.from_pretrained(model_name)
extractor = AutoFeatureExtractor.from_pretrained(model_name)

print("모델과 Feature Extractor가 성공적으로 다운로드되었습니다.")