import numpy as np
import cv2
import glob
import os
# path = 'E:\\10000'
# save_path = 'C:\\Users\\Dell\\Downloads\\yolo_dataset\\train\\images'
# i = 5291
# label_files = glob.glob(path + '/*')
# print(len(label_files))
# for label_file in label_files:
#     i = "%06d" % i
#     img = cv2.imread(label_file)
#     cv2.imwrite(save_path + '/' + str(i) + '.jpg', img)
#     i = int(i) + 1
#     print(label_file)
# print(arr)
# print(arr[0, 1])
path = 'E:/dataset/cornell'
files = glob.glob(path + '/*.txt')
print(len(files))
for file in files:
    if 'Label' not in file:
        print(file)
        os.remove(file)