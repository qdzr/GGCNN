import cv2
import numpy as np
from realsense.realsense_save_rgbd import inpaint

path = 'E:/dataset/changeLabel/pcd1142d.tiff'
# path_file = 'E:/ggcnn/dataset/pos_label/pcd2001d.tiff'
# path_file = '../dataset/clutter/pcd%dd.png'
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
# img.resize((480, 640, 3))
# cv2.imwrite(f'../dataset/changeLabel/test.png', img)
print(img.dtype)
print(img.shape)
cv2.imshow('picture', img)
print(img)
cv2.waitKey(1000)
# 将深度图单位转换到米
# 注意要改两处路径
# for x in range(3264, 3322):
#     cv2_img = cv2.imread(f'../dataset/changeLabel/pcd{x}d.tiff', cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
#     cv2_img = inpaint(cv2_img)  # 补全深度图
#     cv2.imwrite(f'../dataset/changeLabel/pcd{x}d.tiff', cv2_img)
#     cv2.imshow(f"picture", cv2_img)
#     cv2.waitKey(1)
# print('End')

# 读取数据集里的所有彩色图片并显示出来
# i = 1
# for x in range(2000, 2505):
#     cv2_img = cv2.imread(f'../dataset/clutter/pcd{x}r.png', cv2.IMREAD_UNCHANGED)
#     cv2.imshow(f"picture{i}", cv2_img)
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()
#     i = i + 1
# 转变图片命名格式
# for x in range(2000, 2505):
#     cv2_img = cv2.imread(f'../dataset/clutter/pcd{x}d.png', cv2.IMREAD_UNCHANGED)
#     cv2.imwrite(f'../dataset/clutter/pcd{x}d.tiff', cv2_img)
#     cv2_img_rgb = cv2.imread(f'../dataset/clutter/pcd{x}r.png', cv2.IMREAD_UNCHANGED)
#     cv2.imshow("picture", cv2_img_rgb)
