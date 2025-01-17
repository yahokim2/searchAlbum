'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
# python inference_ram_plus_catch1.py --image images/demo/demo3.jpg --pretrained pretrained/ram_plus_swin_large_14m.pth

import argparse
import numpy as np
import random
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from timm.layers import some_layer

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
                    default='pretrained/ram_plus_swin_tiny.pth')  # Swin-Tiny 모델 경로로 수정
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
    # Swin-Tiny 사용으로 변경
    model = ram_plus(pretrained=args.pretrained,
                     image_size=args.image_size,
                     vit='swin_t')  # 'swin_t'는 Swin-Tiny, 'swin_s'는 Swin-Small로 바꿀 수 있음
    model.eval()

    model = model.to(device)

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    res = inference(image, model)

    print("Image Tags: ", res[0])
    print("Image Tags: ", res[1])
