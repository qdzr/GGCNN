import glob
import os
import time
import numpy as np
path = 'E:/dataset/Cornell_grasp_dataset/pos_label2'
save_path = 'E:/dataset/pos_label4'
label_files = glob.glob(os.path.join(path, 'pcd*cpos.txt'))
print(len(label_files))

for label_file in label_files:
    label_points = None
    print(label_file)
    x = []
    y = []
    points = []
    with open(label_file) as f:
        labels = f.readlines()
        for label in labels:
            label = label.strip().split(' ')
            # print(label)
            x.append(float(label[0]))
            y.append(float(label[1]))
        for i in range(0, len(x)):
            points.append((x[i], y[i]))
        print(len(points))
        i1 = 0
        i2 = 1
        i3 = 2
        i4 = 3
        while i1 < len(x):
            point1 = np.hstack(points[i1])
            point2 = np.hstack(points[i2])
            point3 = np.hstack(points[i4])
            point4 = np.hstack(points[i3])
            if label_points is None:
                label_points = point1
                label_points = np.vstack((label_points, point2))
                label_points = np.vstack((label_points, point3))
                label_points = np.vstack((label_points, point4))
            else:
                label_points = np.vstack((label_points, point1))
                label_points = np.vstack((label_points, point2))
                label_points = np.vstack((label_points, point3))
                label_points = np.vstack((label_points, point4))
            i1 += 4
            i2 += 4
            i3 += 4
            i4 += 4
    cpos_file = label_file.replace('pos_label2', 'pos_label5')
    with open(cpos_file, 'w') as p:
        for idx in range(label_points.shape[0]):
            point = label_points[idx]
            for item in point:
                if item != 'None':
                    p.write('{} '.format(item))
            p.write('\n')
print("done")