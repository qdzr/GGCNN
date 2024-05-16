import logging
import shutil

import cv2
import numpy as np
import glob
import os
path = 'E:/dataset/cornell'
save_path = 'E:/dataset/cornell_grey'
img_depth_files = glob.glob(os.path.join(save_path, '*r.png'))
logging.basicConfig(level=logging.INFO)
logging.info('the number of pictures is :{:02d}'.format(len(img_depth_files)))
img = cv2.imread('E:/dataset/cornell_grey/pcd0182d.tiff', -1)
print(img.shape)
img = cv2.resize(img, (300, 300))
print(img.shape)
cv2.imshow('pic', img)
a = np.argmax(img)  # 将矩阵拉成一维的最大值的索引
a_g = np.unravel_index(a, img.shape)  # 最大值索引在原维度的索引坐标 
print("index:", np.argmax(img))
print(a_g)
print(img.squeeze().shape == img.shape)
# for img_depth_file in img_depth_files:
#     img = cv2.imread(img_depth_file, 0).astype(np.float32)/255
#     if not os.path.exists('E:/dataset/cornell_grey'):
#         os.mkdir('E:/dataset/cornell_grey')
#     new_img_path = img_depth_file.replace('r.png', 'd.tiff')
#     cv2.imwrite(new_img_path, img)
    # new_img_path2 = new_img_path.replace('cornell', 'cornell_grey')
    # 复制文件到另一个文件夹
    # shutil.copy(os.path.join(path, new_img_path), os.path.join(save_path, new_img_path2))
    # cv2.imwrite('')
# img = cv2.imread(path, -1).astype(np.float32)/255
# img = cv2.imread(path, 0).astype(np.float32)/255
# print(type(img))
# print(img)
# b, g, r = cv2.split(img)
# cv2.imwrite('data/pcd0001d.tiff', img)
# cv2.imwrite('pcd01120.tiff', r)
# cv2.imshow('picture', img)
cv2.waitKey(1000)
