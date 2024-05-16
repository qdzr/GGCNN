import os
import glob
import scipy.io as scio
import numpy as np
import math

label_path = 'E:/dataset/Cornell_grasp_dataset/image'
save_path = 'E:/dataset/pos_label'


def run():
    label_files = glob.glob(label_path + '/*Label.txt')

    max_w = 0
    for label_file in label_files:
        print('processing', label_file)
        label_points = None
        label_mat = np.zeros((3, 480, 640), dtype=np.float)
        cpos_file = label_file.replace('Label.txt', 'cpos.txt')
        cpos_file = cpos_file.replace('Cornell_grasp_dataset/image', 'pos_label')
        if os.path.exists(cpos_file):
            os.remove(cpos_file)
            # pass
        # if not os.path.exists(cpos_file):
        #     print("no file")
        with open(label_file) as f:
            labels = f.readlines()
            print(len(labels))
            for label in labels:
                x1 = None
                y1 = None
                x2 = None
                y2 = None
                x3 = None
                y3 = None
                x4 = None
                y4 = None
                label = label.strip().split(' ')
                if len(label) == 5:
                    # print(float(label[0])+float(label[4])*math.cos(float(label[2]))/2)
                    x1 = float(label[0]) - float(label[4]) * math.cos(float(label[2])) / 2
                    y1 = float(label[1]) + float(label[4]) * math.sin(float(label[2])) / 2
                    x2 = float(label[0]) + float(label[4]) * math.cos(float(label[2])) / 2
                    y2 = float(label[1]) - float(label[4]) * math.sin(float(label[2])) / 2

                    if abs(x2 - x1) < 0.1:
                        x3 = x1 + 5
                        y3 = y1
                        x4 = x2 + 5
                        y4 = y2
                    else:
                        k = (y2 - y1) / (x2 - x1)

                        if abs(k) <= 0.05:
                            x3 = x1
                            y3 = y1 - 10
                            x4 = x2
                            y4 = y2 - 10
                        else:
                            label_next = (float(label[1]) - 25 / k, float(label[0]) + 25)
                            if k > 0:
                                x3 = x1 - 10
                                y3 = y1 + 10/k
                                x4 = x2 - 10
                                y4 = y2 + 10/k
                            else:
                                x3 = x1 + 10
                                y3 = y1 - 10 / k
                                x4 = x2 + 10
                                y4 = y2 - 10 / k
                            # x3 = float(label_next[1]) - float(label[4]) * math.cos(float(label[2])) / 2
                            # y3 = float(label_next[0]) + float(label[4]) * math.sin(float(label[2])) / 2
                            # x4 = float(label_next[1]) + float(label[4]) * math.cos(float(label[2])) / 2
                            # y4 = float(label_next[0]) - float(label[4]) * math.sin(float(label[2])) / 2

                    # if (x3-x1)**2+(y3-y1)**2 > 12500:
                    #     pass
                    # else:
                    points1 = np.hstack((x1, y1))
                    points2 = np.hstack((x2, y2))
                    #points3 = np.hstack((x3, y3))
                    #points4 = np.hstack((x4, y4))
                    if label_points is None:
                        label_points = points1
                        label_points = np.vstack((label_points, points2))
                        #label_points = np.vstack((label_points, points3))
                        #label_points = np.vstack((label_points, points4))
                    else:
                        label_points = np.vstack((label_points, points1))
                        label_points = np.vstack((label_points, points2))
                        #label_points = np.vstack((label_points, points3))
                        #label_points = np.vstack((label_points, points4))

                if len(label) == 3:
                    for i in np.arange(0, np.pi, 0.1):
                        x1 = float(label[1]) - float(label[2]) * math.cos(i) / 2
                        y1 = float(label[0]) + float(label[2]) * math.sin(i) / 2
                        x2 = float(label[1]) + float(label[2]) * math.cos(i) / 2
                        y2 = float(label[0]) - float(label[2]) * math.sin(i) / 2
                        if abs(x2 - x1) <= 0.01:
                            x3 = x1 + 4
                            y3 = y1
                            x4 = x2 + 4
                            y4 = y2
                        else:
                            k = (y2 - y1) / (x2 - x1)
                            if abs(k) <= 0.05:
                                x3 = x1
                                y3 = y1 + 4
                                x4 = x2
                                y4 = y2 + 4
                            else:
                                label_next = (float(label[0]) - 5 / k, float(label[1]) + 5)
                                if k > 0:
                                    x3 = x1 - 4
                                    y3 = y1 + 4 / k
                                    x4 = x2 - 4
                                    y4 = y2 + 4 / k
                                else:
                                    x3 = x1 + 4
                                    y3 = y1 - 4 / k
                                    x4 = x2 + 4
                                    y4 = y2 - 4 / k
                                # x3 = float(label_next[1]) - float(label[2]) * math.cos(i) / 2
                                # y3 = float(label_next[0]) + float(label[2]) * math.sin(i) / 2
                                # x4 = float(label_next[1]) + float(label[2]) * math.cos(i) / 2
                                # y4 = float(label_next[0]) - float(label[2]) * math.sin(i) / 2
                        if (x3-x1)**2+(y3-y1)**2 > 64:
                            pass
                        else:

                            points1 = np.hstack((x1, y1))
                            points2 = np.hstack((x2, y2))
                            points3 = np.hstack((x3, y3))
                            points4 = np.hstack((x4, y4))
                            if label_points is None:
                                label_points = points1
                                label_points = np.vstack((label_points, points2))
                                label_points = np.vstack((label_points, points3))
                                label_points = np.vstack((label_points, points4))
                            else:
                                label_points = np.vstack((label_points, points1))
                                label_points = np.vstack((label_points, points2))
                                label_points = np.vstack((label_points, points3))
                                label_points = np.vstack((label_points, points4))

            try:
                print("len(label):{}".format(len(label_points)))
            except:
                print("drop data")

        with open(cpos_file, 'w') as p:
            for idx in range(label_points.shape[0]):
                point = label_points[idx]
                for item in point:
                    if item != 'None':
                        p.write('{} '.format(item))
                p.write('\n')

    # print('max_w = ', max_w)
    print("done")


if __name__ == '__main__':
    run()
