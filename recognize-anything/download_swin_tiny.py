import timm
import torch

# Swin-Tiny 모델을 timm 라이브러리에서 로드
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
model.eval()

# GPU가 있을 경우 CUDA로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 예시 입력 이미지
from PIL import Image
from torchvision import transforms

image = Image.open('images/demo/demo3.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_tensor = transform(image).unsqueeze(0).to(device)

# 추론
with torch.no_grad():
    output = model(image_tensor)

print(output)
