import csv
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt
from scale import SCALE_TABLE
from get_the_order import ORDER, QWERT
from tqdm import tqdm

#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = R'C:\Users\tangxiao\Desktop\LAST.pth'
PATH_FLOAT = R'json&raw\YoloV5_normed.raw'                        # 浮点raw路径
PATH_QUANT = R'json&raw\YoloV5_quantized.raw'                     # 定点raw路径
PATH_CSV   = R'logs\quantizer\BUYI\YoloV5\YoloV5_raws.csv'        # csv路径



#-------------------------------------#
#       解析饱和点、Scale
#-------------------------------------#
dic = {}
BIAD = []
with open(PATH_CSV, 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        try:
            raw_id = int(row[0])

            if row[3] == '':
                BIAD.append(raw_id)
                sat_point = 0
                mmax = 0
            else:
                sat_point = float(row[3].split(' ')[0])
                mmax = float(row[2].split(' ')[0])

            scale     = float(row[4].split(' ')[0])
            dic[raw_id] = [sat_point, scale, mmax]
        except:
            continue



#-------------------------------------#
#       解析权重
#-------------------------------------#
# 0-31： 一个int32，表示层数N
# N组，每组包含：一个int32，层ID；一个int32，表示层开始地址
# N层，如果是BIAS，数据为int16；如果是WEIGHT，数据为int8
raw_addr_quant = {}
raw_size_quant = {}
weights_quant  = {}

raw_id   = []


whole_length_quant = len(open(PATH_QUANT, 'rb').read()) # 全部字节数


binfile_quant = open(PATH_QUANT, 'rb')


layer_num = struct.unpack('<i', binfile_quant.read(4))[0]
ANS = struct.pack('i', layer_num)

for i in range(layer_num):

    layer_id   = struct.unpack('<i', binfile_quant.read(4))[0]
    ANS += struct.pack('i', layer_id)

    start_addr_quant = struct.unpack('<i', binfile_quant.read(4))[0]

    raw_addr_quant[layer_id] = start_addr_quant
    ANS += struct.pack('i', start_addr_quant)
    raw_id.append(layer_id)


for i in range(len(raw_id)-1):

    raw_size_quant[raw_id[i]] = raw_addr_quant[raw_id[i+1]] - raw_addr_quant[raw_id[i]]



raw_size_quant[raw_id[i+1]] = whole_length_quant - raw_addr_quant[raw_id[i+1]]


for i in raw_id:
    weights_quant[i] = []

    if i not in BIAD:
        for j in range(0, raw_size_quant[i], 1):
            data = struct.unpack('<b', binfile_quant.read(1))[0]
            weights_quant[i].append(data)
    else:
        for j in range(0, raw_size_quant[i], 2):
            data = struct.unpack('<h', binfile_quant.read(2))[0]
            weights_quant[i].append(data)


WEIGHT = torch.load(PATH_MODEL, map_location=torch.device('cpu'))


#-------------------------------------#
#       画图
#-------------------------------------#
for k in tqdm(weights_quant.keys()):

    WEIGHT[ORDER[k]] = torch.round(WEIGHT[ORDER[k]] / SCALE_TABLE[ORDER[k]])

    try:
        WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]][:,:,[2,1,0],:]

    WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]].flatten().numpy()

    x = WEIGHT[ORDER[k]]
    y = weights_quant[k]

    if k not in BIAD:
        for nnn in WEIGHT[ORDER[k]]:
            ANS += struct.pack('b', int(nnn))
        for i in range(-128, 128, 1):
            plt.axhline(y = i)
    else:
        for nnn in WEIGHT[ORDER[k]]:
            ANS += struct.pack('h', int(nnn))


with open('json&raw/YoloV5_quantized_INQ.raw', 'ab+') as f:
    f.write(ANS)
    f.close()