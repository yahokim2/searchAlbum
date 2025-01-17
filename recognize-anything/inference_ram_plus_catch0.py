'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
# python inference_ram_plus_catch0.py --image images/demo/demo3.jpg --pretrained pretrained/ram_plus_swin_large_14m.pth

import argparse
import numpy as np
import random
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

import os  # os 모듈 추가

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

    # tags = res[0].split(' | ')  # '|' 으로 구분된 태그들

    # person = 0
    # nop = 0

    # for word in tags:
    #     if word.lower() in ['man', 'woman', 'girl', 'boy']:
    #         person += 1
    #     else:
    #         nop += 1
    # print(f"Count of 'man' or 'woman' or 'girl' or 'boy' : {person}")
    # print(f"Count of things : {nop}")

    # if person == 0:
    #     play_audio("hello_empty.mp3")  

    # if person == 1:
    #     play_audio("hello_a1.mp3")          
    #     play_audio("hello_a2.mp3") 

    # elif person > 1 and person <= 5:
    #     play_audio("hello_b1.mp3")  
    #     play_audio("hello_b2.mp3")