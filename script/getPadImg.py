"""
把测试图片提前resize到640*640, 通过添加灰边来保持长宽比不变

Usage:
    $ python .\script\getPadImg.py
"""

import os
import cv2
from tqdm import tqdm



#-------------------------------------#
#       文件路径
#-------------------------------------#
IN_PATH  = 'assets/val2017/'
OUT_PATH = 'assets/val640/'

if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)



#-------------------------------------#
#       遍历测试集
#-------------------------------------#
for file in tqdm(os.listdir(IN_PATH)):

    img = cv2.imread(IN_PATH + file)

    img_h = img.shape[0]
    img_w = img.shape[1]
    
    ratio = min(640 / img_w, 640 / img_h)

    w_unpad = int(round(img_w * ratio))
    h_unpad = int(round(img_h * ratio))

    resized_img = cv2.resize(img, (w_unpad, h_unpad), 0, 0, cv2.INTER_LINEAR)
    resized_img = cv2.copyMakeBorder(resized_img, 0, abs(640 - h_unpad), 0, abs(640 - w_unpad), cv2.BORDER_CONSTANT, value= (144, 144, 144))

    cv2.imwrite(OUT_PATH + file, resized_img)
