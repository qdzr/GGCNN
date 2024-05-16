# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/22 19:50
@Auth ： 王德鑫
@File ：learn.py
@IDE ：PyCharm
@Function: 将拍摄的图像文件名修改为pcd...r.png
"""

import os
import glob
import shutil 

path = 'E:/dataset/data'

rgb_files = glob.glob(os.path.join(path, '*.png'))
rgb_files.sort()

i = 3396
for rgb_file in rgb_files:
    depth_file = rgb_file.replace('r.png', 'd.tiff')

    new_rgb_name = 'pcd{:04d}r.png'.format(i)
    new_rgb_file = os.path.join(path, new_rgb_name)
    shutil.copyfile(rgb_file, new_rgb_file)

    new_dep_name = 'pcd{:04d}d.tiff'.format(i)
    new_dep_file = os.path.join(path, new_dep_name)
    shutil.copyfile(depth_file, new_dep_file)

    i += 1

    print(new_rgb_file)

