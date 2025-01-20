from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 모델과 프로세서 불러오기
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 이미지 로드
image = Image.open("images/demo/demo1.jpg")

# 이미지 캡션 생성
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print(caption)
