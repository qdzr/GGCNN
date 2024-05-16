import torch
import numpy as np
import cv2
import glob
from utils.ggcnn import GGCNN, input_img
import os
import math
import time
import glob


def drawGrasps(img, grasps, mode):
    """
    绘制grasp
    img: img data
    grasps: list()	元素是 [row, col, angle, width]
    mode: line / region
    """
    assert mode in ['line', 'region']

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp
        print('width:{}'.format(width))

        if mode == 'line':
            width = width / 2
            k = math.tan(angle)
            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            cv2.line(img, (int(col + dx), int(row - dy)), (int(col - dx), int(row + dy)), (0, 0, 255), 1)

            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
            return (col + dx, row - dy), (col - dx, row + dy)

        else:
            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            img[row, col] = [color_b, color_g, color_r]


def resize2ori(resize_shape, ori_shape, resize_pixel):
    h1, w1 = ori_shape
    # print(h1, w1)
    h2, w2 = resize_shape
    # print(h2, w2)
    y1 = resize_pixel[1] * h1 / h2
    x1 = resize_pixel[0] * w1 / w2
    return x1, y1


def resize_draw(crop_color, crop_img_resize):  # resize_to_ori不能把resize后的像素恢复原样
    cv2.circle(crop_color, (grasps[0][1], grasps[0][0]), 7, (0, 255, 0), -1)
    crop_reset = cv2.resize(crop_color, (crop_img.shape[1], crop_img.shape[0]))

    print(grasps)
    pixel = [grasps[0][1], grasps[0][0]]
    p1, p2 = resize2ori(crop_img_resize.shape, crop_img.shape, pixel)
    cv2.circle(image, (int(p1) + 150, (int(p2) + 80)), 3, (0, 255, 0), -1)
    point1, point2 = drawGrasps(crop_color, grasps, mode='line')
    print(point1, point2)
    p11, p12 = resize2ori(crop_img_resize.shape, crop_img.shape, point1)
    p21, p22 = resize2ori(crop_img_resize.shape, crop_img.shape, point2)
    cv2.circle(image, (int(p11) + 150, (int(p12) + 80)), 3, (0, 255, 0), -1)
    cv2.circle(image, (int(p21) + 150, (int(p22) + 80)), 3, (0, 255, 0), -1)
    cv2.line(image, (int(p11) + 150, (int(p12) + 80)), (int(p21) + 150, (int(p22) + 80)), (255, 0, 0), 1)
    time_b = time.time()
    print('spend time: ', (time_b - time_a) * 1000, 'ms')
    cv2.imshow('crop_reset', crop_reset)
    cv2.imshow('crop_color', crop_color)
    cv2.imshow('image', image)
    cv2.waitKey(10000)


# 模型路径
model = 'ckpt/epoch_0282_acc_0.6705.pth'
input_path = 'data/pcd0100d.tiff'
color_path = 'data/pcd0100r.png'

depth_img_files = glob.glob('data/' + '*d.tiff')
print(depth_img_files)
# 运行设备
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
# 初始化
ggcnn = GGCNN(model, device=device_name)

for depth_img in depth_img_files:
    time_a = time.time()
    img_depth = cv2.imread(depth_img, -1)
    crop_img = img_depth[260:380, 180:300]
    print(crop_img.shape)
    cv2.imshow('crop_img_1', crop_img)
    pad_up = int((360 - (380 - 260)) / 2)
    pad_left = int((360 - (300 - 180)) / 2)
    # constant_value = np.mean(crop_img)
    # constant_value = np.max(crop_img)
    min = np.min(crop_img)
    max = np.max(crop_img)
    constant_value = np.mean(crop_img[0:6, 0:6])
    print('constant_value: ', constant_value)
    crop_img = np.pad(crop_img, ((pad_up, pad_up), (pad_left, pad_left)), 'constant', constant_values=(constant_value, constant_value))
    print(crop_img.shape)
    print(crop_img)
    cv2.imshow('crop_img_pad', crop_img)

    rgb_img = depth_img.replace('d.tiff', 'r.png')
    image = cv2.imread(rgb_img)
    crop_color = image[260:380, 180:300]
    crop_color = np.pad(crop_color, ((pad_up, pad_up), (pad_left, pad_left), (0, 0)), 'constant', constant_values=(255, 255))
    cv2.imshow('crop_color', crop_color)
    print(crop_color.shape)

    time_0 = time.time()
    grasps, x1, y1 = ggcnn.predict(crop_img, mode='max')
    point1, point2 = drawGrasps(crop_color, grasps, mode='line')
    time_1 = time.time()
    cv2.circle(crop_color, (grasps[0][1], grasps[0][0]), 2, (255, 0, 0), -1)
    print('predict time: ', (time_1 - time_0) * 1000, 'ms')
    print('grasp: ', grasps)
    cv2.imshow('draw', crop_color)
    cv2.waitKey(10000)
