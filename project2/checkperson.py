'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
 * Modified for saving captured frames at specified intervals 
'''
# command ex) python inference_ram_plus_webcam2.py --pretrained pretrained/ram_plus_swin_large_14m.pth --interval 20 

import argparse
import numpy as np
import random
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
#from gtts import gTTS
#import pygame
import os
import cv2
import time
import csv  # CSV 모듈 추가

# CSV 파일에 헤더 추가
def write_csv_header(csv_file):
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Time', 'Person Count'])  # 헤더 작성

# CSV에 데이터 추가
def write_to_csv(csv_file, date, current_time, person_cnt):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([date, current_time, person_cnt])  # 날짜, 시간, 사람 수 기록

def generate_audio_files():
#   a1 = "안녕하세요! AI 헬스장입니다."     
    a1 = "안녕하세요"
#   a2 = "좀 더 카메라 앞으로 가까이 와 주세요."
    a2 = "한 분! 가까이오세요"

#   b1 = "어서오세요 여러분! 여기는 자세 도우미가 있는 AI 헬스장입니다."
    b1 = "어서오세요!"
#   b2 = "오시는 순서대로 카메라 앞으로 다가와 주세요."
    b2 = "여러분"


    emp99 = "지금은 아무도 없습니다."

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

# 웹캠 입력을 받아 주기마다 이미지를 처리하고 결과 출력하는 함수
def process_webcam_input(model, device, transform, interval, csv_file):
    cap = cv2.VideoCapture(0)  # 웹캠에서 영상 캡처
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    # 웹캠에서 지정된 주기마다 이미지를 캡처하여 판단
    last_processed_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # 현재 시간
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # 주기마다 이미지 처리
        if current_time - last_processed_time > interval:
            last_processed_time = current_time

            # 웹캠에서 캡처한 이미지를 PIL 이미지로 변환
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 이미지를 변환
            input_tensor = transform(image).unsqueeze(0).to(device)

            # 모델을 이용한 추론
            res = inference(input_tensor, model)
            print("Image Tags: ", res[0])
            print("Image Tags: ", res[1])

            tags = res[0].split(' | ')  # '|' 으로 구분된 태그들

            person_cnt = 0
            nop = 0
            for word in tags:
                if word.lower() in ['man', 'woman', 'girl', 'boy']:
                    person_cnt += 1
                else:
                    nop += 1
            print(f"Count of 'man' or 'woman' or 'girl' or 'boy' : {person_cnt}")   

            # 인원 수에 따라 음성 파일 재생
            if person_cnt == 0:
                play_audio("hello_empty.mp3")  

            if person_cnt == 1:
                play_audio("hello_a1.mp3")          
                play_audio("hello_a2.mp3") 

            elif person_cnt > 1 and person_cnt <= 3:
                play_audio("hello_b1.mp3")  
                play_audio("hello_b2.mp3")        

            # 이미지 저장 (파일명은 시간 기반으로)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_filename = f"captured_image_{timestamp}.jpg"
            cv2.imwrite(img_filename, frame)  # 이미지 파일로 저장
            print(f"이미지가 저장되었습니다: {img_filename}")

            # 현재 날짜와 시간을 가져오기
            date = time.strftime("%Y-%m-%d")
            current_time_str = time.strftime("%H:%M:%S")
            
            # CSV 파일에 기록
            write_to_csv(csv_file, date, current_time_str, person_cnt)
        
        # 캡처된 이미지를 화면에 표시
        cv2.imshow("Webcam", frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 웹캠 자원 해제
    cv2.destroyAllWindows()  # 창 닫기

parser = argparse.ArgumentParser(
    description='Tag2Text inference for tagging and captioning')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')
parser.add_argument('--interval',
                    default=20,
                    type=int,
                    metavar='N',
                    help='interval in seconds between each prediction (default: 20)')
parser.add_argument('--csv-file',
                    default='captured_data.csv',
                    type=str,
                    help='CSV file to store the captured data')

if __name__ == "__main__":

    # 음성 파일 생성
    generate_audio_files()

    # arguments
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    ####### load model
    model = ram_plus(pretrained=args.pretrained,
                     image_size=args.image_size,
                     vit='swin_l')
    model.eval()

    model = model.to(device)

    # CSV 파일에 헤더 추가
    write_csv_header(args.csv_file)

    # 웹캠 입력 받아 주기마다 이미지 처리
    process_webcam_input(model, device, transform, args.interval, args.csv_file)
