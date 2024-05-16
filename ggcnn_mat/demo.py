"""
Description:
Author: wangdx
Date: 2021-11-28 13:47:41
LastEditTime: 2021-11-28 14:33:57
"""

import cv2
import os
import torch
import math
import glob
from utils.ggcnn import GGCNN
import numpy as np


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
        # as_gr = rectangle_points([row, col], angle, width)
        # cv2.line(img_rgb, (as_gr[0][1], as_gr[0][0]), (as_gr[1][1], as_gr[1][0]), (255, 0, 0), 1)
        # cv2.line(img_rgb, (as_gr[2][1], as_gr[2][0]), (as_gr[3][1], as_gr[3][0]), (255, 0, 0), 1)
        # cv2.line(img_rgb, (as_gr[3][1], as_gr[3][0]), (as_gr[0][1], as_gr[0][0]), (0, 0, 255), 1)
        # cv2.line(img_rgb, (as_gr[1][1], as_gr[1][0]), (as_gr[2][1], as_gr[2][0]), (0, 0, 255), 1)

        if mode == 'line':
            width = width / 2
            k = math.tan(angle)
            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            cv2.line(img, (int(col + dx), int(row - dy)), (int(col - dx), int(row + dy)), (0, 255, 0), 1)

            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)

        else:
            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            img[row, col] = [color_b, color_g, color_r]


def drawRect(img, rect):
    """
    绘制矩形
    rect: [x1, y1, x2, y2]
    """
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


def rectangle_points(center, angle, length):
    """
    Convert to GraspRectangle
    :return: GraspRectangle representation of grasp.
    """
    xo = np.cos(angle)
    yo = np.sin(angle)
    width = length / 2

    y1 = center[0] + length / 2 * yo
    x1 = center[1] - length / 2 * xo
    y2 = center[0] - length / 2 * yo
    x2 = center[1] + length / 2 * xo

    return np.array(
        [
            [y1 - width / 2 * xo, x1 - width / 2 * yo],
            [y2 - width / 2 * xo, x2 - width / 2 * yo],
            [y2 + width / 2 * xo, x2 + width / 2 * yo],
            [y1 + width / 2 * xo, x1 + width / 2 * yo],
        ]
    ).astype(np.int)


if __name__ == '__main__':
    # 模型路径
    model = 'ckpt/epoch_0282_acc_0.6705.pth'
    input_path = 'E:/dataset/cornell'

    # 运行设备
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    # 初始化
    ggcnn = GGCNN(model, device=device_name)

    img_depth_files = glob.glob(os.path.join(input_path, '*d.tiff'))

    for img_depth_file in img_depth_files:
        print('processing ', img_depth_file)
        # 读取图像
        img_depth = cv2.imread(img_depth_file, -1)
        rgb_file = img_depth_file.replace('d.tiff', 'r.png')
        # rgb_file = rgb_file.replace('pos_label4', 'image')
        img_rgb = cv2.imread(rgb_file)
        # 预测
        grasps, x1, y1 = ggcnn.predict(img_depth, mode='max')
        # print(grasps)
        # 绘制预测结果
        drawGrasps(img_rgb, grasps, mode='line')
        rect = [x1, y1, x1 + 300, y1 + 300]
        # drawRect(img_rgb, rect)

        # 可视化
        cv2.imshow('grasp', img_rgb)
        # rgb_file = img_depth_file.replace('data', 'data3')
        # cv2.imwrite(rgb_file, img_rgb)
        cv2.waitKey(100)

    print('FPS: ', ggcnn.fps())
