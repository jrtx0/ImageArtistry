# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from PIL import Image

# 图片的输入和输出路径
imgInput = "/Users/jrtx/Pictures/input/"
imgOutput = "/Users/jrtx/Pictures/output/"

def cartoonise(picture_name):
    imgInput_FileName = imgInput + picture_name
    print(imgInput_FileName)
    imgOutput_FileName = imgOutput + 'cartoonise_' + picture_name
    num_down = 2 # 缩减像素采样的数目
    num_bilateral = 7 # 定义双滤边的数目
    img_rgb = cv2.imread(imgInput_FileName) # 读取图片
    shape = tuple(list(((i // 4) * 4 for i in img_rgb.shape))[:2])[::-1]
    img_rgb = cv2.resize(img_rgb, dsize=shape)

    # 用高斯金字塔降低取样
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    # 重复使用小的双边滤波代替一个大的滤波
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    # 升采样图片到原始大小
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    # 转换为灰度并且使其产生中等的模糊
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    # 检测到边缘并且增强其效果
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    # 转换回彩色图像
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_carton = cv2.bitwise_and(img_color, img_edge)
    # 保存转换后的图片
    cv2.imwrite(imgOutput_FileName, img_edge)

def handpainted(picture_name):
    imgInput_FileName = imgInput + picture_name
    imgOutput_FileName = imgOutput + 'handpainted_' + picture_name
    a = np.array(Image.open(imgInput_FileName).convert('L')).astype('float')
    img_depth = 10
    img_grad = np.gradient(a)
    grad_x, grad_y = img_grad
    grad_x = grad_x * img_depth / 100
    grad_y = grad_y * img_depth / 100
    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1. / A

    vec_el = np.pi / 2.2
    vec_ez = np.pi / 4.
    dx = np.cos(vec_el) * np.cos(vec_ez)
    dy = np.cos(vec_el) * np.sin(vec_ez)
    dz = np.sin(vec_el)

    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
    b = b.clip(0, 255)

    im = Image.fromarray(b.astype('uint8'))
    # 保存后转化的图片
    im.save(imgOutput_FileName)


ImageList = []
# 循环读取目录中的文件名
for filename in os.listdir(imgInput):
    if (filename != '.DS_Store'):
        ImageList.append(filename)

for i in ImageList:
    cartoonise(i)
    handpainted(i)