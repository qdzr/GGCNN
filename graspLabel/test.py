import cv2
import glob
import os
import copy
import time
import math
import numpy as np
import matplotlib.pyplot as plt

path = 'E:/dataset/Cornell_grasp_dataset/pos_label4'     # 数据集路径
# path = 'E:/dataset/test'  # 数据集路径
# path = 'E:/dataset/pos_label3'
savepath = 'E:/dataset/test/show_label'
label_win_name = 'labeling'
# label_files = glob.glob(os.path.join(path, 'pcd*Label.txt'))
label_files = glob.glob(os.path.join(path, 'pcd*cpos.txt'))
print(len(label_files))
# print(np.arange(0, 1, 0.1))

for labelfile in label_files:
    i = 0
    x = []
    y = []
    pngfile = labelfile.replace('cpos.txt', 'r.png')
    # pngfile = pngfile.replace('pos_label3', 'Cornell_grasp_dataset/image')
    pngfile = pngfile.replace('pos_label4', 'image')
    print(pngfile)
    if not os.path.exists(pngfile):
        continue
    im = cv2.imread(pngfile)
    with open(labelfile) as f:
        labels = f.readlines()
        for label in labels:
            label = label.strip().split(' ')
            # print(label)
            i = i + 1
            x.append(int(float(label[0])))
            y.append(int(float(label[1])))
            # if i % 4 == 0:
            #     break
        if len(x) == len(y):
            print('len: %d' % len(x))
    # cv2.rectangle(im, (286, 257), (277, 248), (0, 255, 0), 2)
    # cv2.line(im,  (253, 320), (309, 324), (0, 255, 0), 1)
    # cv2.line(im, (307, 350), (251, 346), (0, 255, 0), 1)
    # cv2.line(im, (255, 325), (308, 332), (0, 255, 0), 1)
    # cv2.line(im, (313, 295), (260, 288), (0, 255, 0), 1)
    # pngfile = pngfile.replace('redwall_single', 'test/show_label')
    # i=4
    # lens = len(x) / 4
    # for len in range(0, int(lens)):
    len1 = 0
    while len1 < len(x):
        cv2.line(im, (x[len1], y[len1]), (x[len1 + 1], y[len1 + 1]), (255, 0, 0), 1)
        cv2.line(im, (x[len1 + 2], y[len1 + 2]), (x[len1 + 3], y[len1 + 3]), (255, 0, 0), 1)
        cv2.line(im, (x[len1], y[len1]), (x[len1 + 3], y[len1 + 3]), (0, 255, 0), 1)
        cv2.line(im, (x[len1 + 1], y[len1 + 1]), (x[len1 + 2], y[len1 + 2]), (0, 255, 0), 1)
        len1 += 4
    cv2.imshow('im', im)
    cv2.waitKey(10000)

    # cv2.waitKey(1)
    # pngfile = pngfile.replace('r.png', 't.png')
    # cv2.imwrite(pngfile, im)
        # if len > 1000:
        #     break
    # plt.imshow(im[:,:-1])
    # plt.show()


