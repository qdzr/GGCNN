# -*- coding:utf-8 -*-
'''
test.py
'''

import cv2
import random
import argparse

clicked = False
g_rectangle = [0, 0, 0, 0]
g_startPoint = [0, 0]


def onMouse(event, x, y, flags, param):
    global clicked
    global g_rectangle
    global g_startPoint
    if event == cv2.EVENT_MOUSEMOVE:
        if clicked == True:
            g_rectangle[0] = min(g_startPoint[0], x)
            g_rectangle[1] = min(g_startPoint[1], y)
            g_rectangle[2] = max(g_startPoint[0], x)
            g_rectangle[3] = max(g_startPoint[1], y)

    # 左键按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        g_startPoint[0] = x
        g_startPoint[1] = y
        clicked = True
    # 左键弹起事件
    if event == cv2.EVENT_LBUTTONUP:
        print("====================选中框的坐标：===========================")
        print("矩形框左上角坐标：")
        print(g_rectangle[0], g_rectangle[1])
        print("矩形框右下角坐标：")
        print(g_rectangle[2], g_rectangle[3])
        clicked = False


def startRoi(path):
    cv2.namedWindow("MyWindow", 0)
    cv2.resizeWindow("MyWindow", 1280, 720)  # 设置长和宽
    cv2.setMouseCallback("MyWindow", onMouse)

    # 按"Esc"退出
    print("Press Esc if you want to exit ...")
    while cv2.waitKey(30) != 27:
        global frame
        frame = cv2.imread(path)
        # 画矩形
        cv2.rectangle(frame, (g_rectangle[0], g_rectangle[1]), (g_rectangle[2], g_rectangle[3]), (0, 255, 0), 2)
        cv2.imshow("MyWindow", frame)

    cv2.destroyWindow("MyWindow")


if __name__ == '__main__':
    path = r"E:\graspLabel\test\pcd0100r.png"
    startRoi(path)
