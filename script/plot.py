import torch
import numpy as np
from get_the_order import QWERT
import matplotlib.pyplot as plt
from utils import get_scale_dict



#-------------------------------------#
#       文件路径
#-------------------------------------#
BEFORE_INQ = 'weights/quantize.pth'
AFTER_INQ  = 'weights/INQ99.pth'
PATH_CSV   = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'



#-------------------------------------#
#       解析Scale
#-------------------------------------#
SCALE_TABLE = get_scale_dict(PATH_CSV)



#-------------------------------------#
#       画图
#-------------------------------------#
a = torch.load(BEFORE_INQ, map_location = torch.device('cpu'))
b = torch.load(AFTER_INQ , map_location = torch.device('cpu'))

for key in a.keys():

    x = a[key].cpu().flatten().numpy() / SCALE_TABLE[QWERT[key]]
    y = b[key].cpu().flatten().numpy() / SCALE_TABLE[QWERT[key]]

    # 50% 分界线
    I = np.quantile(abs(y), 1 - 0.5)
    plt.axhline(y = -I)
    plt.axhline(y =  I)

    for i in range(len(x)):
        if x[i] != y[i]:
            plt.plot([i, i], [min(x[i], y[i]), max(x[i], y[i])], linestyle='--', color = 'black')

    # 整数线
    if 'weight' in key:
        for i in range(-128, 127, 1):
            plt.axhline(y = i, alpha = 1/5, color = 'black')

    plt.plot(x, '.', color = 'red'  , alpha = 1/2)
    plt.plot(y, '.', color = 'green', alpha = 1/2)

    plt.show()
    plt.cla()
