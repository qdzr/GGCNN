# -*- coding: utf-8 -*-
import os
import glob
# 设定文件路径
path = 'E:/dataset/redwall_single/data'
save_path = "E:/dataset/redwall_single"
label_files = glob.glob(os.path.join(path, '*.png'))
i = 3322
for label_file in label_files:
    rgb_new_name = label_file.replace(label_file, "pcd%dr.png" % i)
    os.rename(os.path.join(path, label_file), os.path.join(save_path, rgb_new_name))
    depth_file = label_file.replace('r.png', 'd.tiff')
    if os.path.isfile(os.path.join(path, depth_file)):
        print(depth_file)
        new_name = depth_file.replace(depth_file, "pcd%dd.tiff" % i)
        os.rename(os.path.join(path, depth_file), os.path.join(save_path, new_name))
        i += 1
    if not os.path.exists(label_file):
        continue
# depth_files = glob.glob(os.path.join(path, '*.png'))
# print(len(depth_files))
# i = 1132
# for labelfile in label_files:
#     if os.path.isfile(os.path.join(path, labelfile)):
#         new_name = labelfile.replace(labelfile, "pcd%dr.png" % i)
#         os.rename(os.path.join(path, labelfile), os.path.join(path, new_name))
#         i += 1
#     if not os.path.exists(labelfile):
#         continue
# i = 1132
# for depth_file in depth_files:
#     if os.path.isfile(os.path.join(path, depth_file)):
#         depth_new_name = depth_file.replace(depth_file, "pcd%dd.tiff" % i)
#         os.rename(os.path.join(path, depth_file), os.path.join(path, depth_new_name))
#         i += 1
#         if not os.path.exists(depth_file):
#             continue
# 对目录下的文件进行遍历
# for file in os.listdir(path):
#     # 判断是否是文件
#     if os.path.isfile(os.path.join(path, file)) == True:
#         # 设置新文件名
#         new_name = file.replace(file, "rgb_%d.jpg" % i)
#         # 重命名
#         os.rename(os.path.join(path, file), os.path.join(path, new_name))
#         i += 1
# 结束
print("End")
