'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
# python inference_ram_plus_ponly0.py --image images/demo/demo5.jpg --pretrained pretrained/ram_plus_swin_large_14m.pth

import argparse
import numpy as np
import random
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from gtts import gTTS
import pygame
import os  # os 모듈 추가

def generate_audio_files():
    a1 = "안녕하세요! AI 헬스장입니다."
    a2 = "좀 더 카메라 앞으로 가까이 와 주세요."
    b1 = "어서오세요 여러분! 여기는 자세 도우미가 있는 AI 헬스장입니다."
    b2 = "앞에 오시는 순서대로 카메라 앞으로 다가와 주세요."
    emp99 = "지금은 오시는 회원이 한 분도 없습니다."

    # 각 음성 파일 이름 리스트
    audio_files = [
        ("hello_a1.mp3", a1),
        ("hello_a2.mp3", a2),
        ("hello_b1.mp3", b1),
        ("hello_b2.mp3", b2),
        ("hello_empty.mp3", emp99)
    ]
    
    # 음성 파일을 생성하기 전에 해당 파일이 존재하는지 확인
    for filename, text in audio_files:
        if os.path.exists(filename):
            print(f"이미 음성 파일 '{filename}'이(가) 존재합니다.")
        else:
            tts = gTTS(text=text, lang='ko')
            tts.save(filename)
            print(f"음성 파일 '{filename}'이 생성되었습니다.")

# 음성을 재생하는 함수 정의
def play_audio(audio_file):
    """주어진 음성 파일을 재생하는 함수."""
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # 음성 재생 중일 때
        pygame.time.Clock().tick(10)
    print(f"음성 '{audio_file}' 재생이 끝났습니다.")

parser = argparse.ArgumentParser(
    description='Tag2Text inference for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/demo/demo1.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')

if __name__ == "__main__":

    # 음성 파일 생성
    generate_audio_files()

    # arguments
    args = parser.parse_args()

    ##### 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    ####### load model
    model = ram_plus(pretrained=args.pretrained,
                     image_size=args.image_size,
                     vit='swin_l')
    model.eval()

    model = model.to(device)

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    res = inference(image, model)

    print("Image Tags: ", res[0])
    print("Image Tags: ", res[1])

    tags = res[0].split(' | ')  # '|' 으로 구분된 태그들

    person = 0
    nop = 0

    for word in tags:
        if word.lower() in ['man', 'woman', 'girl', 'boy']:
            person += 1
        else:
            nop += 1
    print(f"Count of 'man' or 'woman' or 'girl' or 'boy' : {person}")
    print(f"Count of things : {nop}")

    if person == 0:
        play_audio("hello_empty.mp3")  

    if person == 1:
        play_audio("hello_a1.mp3")          
        play_audio("hello_a2.mp3") 

    elif person > 1 and person <= 5:
        play_audio("hello_b1.mp3")  
        play_audio("hello_b2.mp3")