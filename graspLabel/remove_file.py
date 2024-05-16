# Importing the os library
import os
import cv2
import matplotlib.pyplot as plt
import glob
path = 'E:/dataset/Cornell_grasp_dataset/depth_image'
label_files = glob.glob(os.path.join(path, 'pcd*.txt'))
print(len(label_files))
for labelfile in label_files:
    os.remove(labelfile)
    if not os.path.exists(labelfile):
        continue
# Inbuilt function to remove files
# 移除路径内所有grasp.mat文件
# for x in range(2001, 2505):
#     os.remove(f"E:/dataset/clutter/pcd{x}grasp.mat")
#
# print("File removed successfully")

# img = cv2.imread('test/pcd0114r.png')
# rows, cols = img.shape[: 2]
# print(rows)
# plt.imshow(img[:, :, ::-1])
# print(img.shape)
# plt.show()