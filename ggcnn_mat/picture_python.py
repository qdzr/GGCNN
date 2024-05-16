import re
import time
import requests
from bs4 import BeautifulSoup
import os
import glob
import cv2
import shutil

# m = 'https://tse2-mm.cn.bing.net/th/id/OIP-C.uihwmxDdgfK4FlCIXx-3jgHaPc?w=115&amp;h=183&amp;c=7&amp;r=0&amp;o=5&amp;pid=1.7'
'''
resp = requests.get(m)
byte = resp.content
print(os.getcwd())
img_path = os.path.join(m)
'''


def main(baseurl):
    # baseurl = 'https://cn.bing.com/images/search?q=%E6%83%85%E7%BB%AA%E5%9B%BE%E7%89%87&qpvt=%e6%83%85%e7%bb%aa%e5%9b%be%e7%89%87&form=IGRE&first=1&cw=418&ch=652&tsc=ImageBasicHover'
    # baseurl = 'https://cn.bing.com/images/search?q=%e9%a5%ae%e6%96%99%e7%93%b6&qpvt=%e9%a5%ae%e6%96%99%e7%93%b6&form=IGRE&first=1&tsc=ImageHoverTitle'
    datalist = getdata(baseurl)


def getdata(baseurl):
    Img = re.compile(r'img.*src="(.*?)"')  # 正则表达式匹配图片
    datalist = []
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"}
    response = requests.get(baseurl, headers=head)  # 获取网页信息
    html = response.text  # 将网页信息转化为text形式
    soup = BeautifulSoup(html, "html.parser")  # BeautifulSoup解析html
    # i = 0  # 计数器初始值
    data = []  # 存储图片超链接的列表
    for item in soup.find_all('img', src=""):  # soup.find_all对网页中的img—src进行迭代
        item = str(item)  # 转换为str类型
        Picture = re.findall(Img, item)  # 结合re正则表达式和BeautifulSoup, 仅返回超链接
        for b in Picture:  # 遍历列表，取最后一次结果
            data.append(b)
            # i = i + 1
            datalist.append(data[-1])
    return datalist  # 返回一个包含超链接的新列表
    # print(i)


'''
with open("img_path.jpg","wb") as f:
    f.write(byte)
'''

if __name__ == '__main__':
    path = 'E:/'
    write_path = 'E:/10000'
    nums = 22
    s = 5523
    need_del = False
    save_path = path + str(32)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.chdir(save_path)
    baseurl = 'https://cn.bing.com/images/search?q=%E5%A5%A5%E5%88%A9%E5%A5%A5%E9%A5%BC%E5%B9%B2%E7%9B%926&qs=n&form=QBIR&sp=-1&pq=%E5%A5%A5%E5%88%A9%E5%A5%A5%E9%A5%BC%E5%B9%B2%E7%9B%926&sc=0-7&cvid=3F72A6B6D0D4479BB46A356A074E5234&ghsh=0&ghacc=0&first=1&tsc=ImageHoverTitle'
    main(baseurl)
    i = 0  # 图片名递增
    # for m in getdata(
    #         # baseurl='https://cn.bing.com/images/search?q=%E6%83%85%E7%BB%AA%E5%9B%BE%E7%89%87&qpvt=%e6%83%85%e7%bb%aa%e5%9b%be%e7%89%87&form=IGRE&first=1&cw=418&ch=652&tsc=ImageBasicHover'):
    #         baseurl):
    #     break
    #     resp = requests.get(m)  # 获取网页信息
    #     byte = resp.content  # 转化为content二进制
    #     print(os.getcwd())  # os库中输出当前的路径
    #     i = i + 1  # 递增
    #     # img_path = os.path.join(m)
    #     with open("path{}.jpg".format(i), "wb") as f:  # 文件写入
    #         f.write(byte)
    #         time.sleep(0.5)  # 每隔0.5秒下载一张图片放入D://情绪图片测试
    #     print("第{}张图片爬取成功!".format(i))

    for j in range(1, nums):
        if need_del:
            break
        save_path = path + str(j)
        os.chdir(save_path)
        imgs = glob.glob(save_path + '/*')
        for img in imgs:
            s = "%06d" % s
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            pic = cv2.imread(img)
            cv2.imwrite(write_path + '/' + str(s) + '.jpg', pic)
            s = int(s) + 1

    for j in range(1, nums):
        if not need_del:
            break
        save_path = path + str(j)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)