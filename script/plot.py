import torch
import argparse
from utils import smart_load
import matplotlib.pyplot as plt



#-------------------------------------#
#       文件路径
#-------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--PATH_MODEL0', type = str, default = 'weights/norm.pth')
parser.add_argument('--PATH_MODEL1', type = str, default = 'weights/quantize.pth')
parser.add_argument('--ratio', type = float, default = None)
opt = parser.parse_args()



#-------------------------------------#
#       解析权重
#-------------------------------------#
weights0 = smart_load(opt.PATH_MODEL0)
weights1 = smart_load(opt.PATH_MODEL1)



#-------------------------------------#
#       画图
#-------------------------------------#
for k in weights0.keys():

    x = weights0[k].flatten()
    y = weights1[k].flatten()

    if opt.ratio:
        I = torch.quantile(torch.abs(x).flatten(), 1 - opt.ratio)
        plt.axhline(y =  I)
        plt.axhline(y = -I)

    for i in range(len(x)):
        if x[i] != y[i]:
            plt.plot([i, i], [min(x[i], y[i]), max(x[i], y[i])], linestyle = 'dotted', alpha = 0.5, color = 'black')

    plt.plot(x, '.', alpha = 0.5, color = 'red')
    plt.plot(y, '.', alpha = 0.5, color = 'green')

    plt.show()
    plt.cla()
