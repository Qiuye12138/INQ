import torch
import struct
import argparse
from get_the_order import ORDER
from utils import *



#-------------------------------------#
#       文件路径
#-------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--PATH_MODEL', type = str, default = 'weights/INQ99.torchscript')
parser.add_argument('--PATH_QUANT', type = str, default = 'json&raw/YoloV5_quantized.raw')
parser.add_argument('--PATH_CSV'  , type = str, default = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv')
opt = parser.parse_args()



#-------------------------------------#
#       解析Scale
#-------------------------------------#
SCALE_TABLE = get_scale_dict(opt.PATH_CSV)



#-------------------------------------#
#       解析权重
#-------------------------------------#
binfile_quant = open(opt.PATH_QUANT, 'rb')
layer_num = struct.unpack('<i', binfile_quant.read(4))[0]
ANS = struct.pack('i', layer_num)
ANS += binfile_quant.read(layer_num * 2 * 4)

weights_quant = bin2numpy_int16(opt.PATH_QUANT, opt.PATH_CSV)

WEIGHT = smart_load(opt.PATH_MODEL)



#-------------------------------------#
#       转换
#-------------------------------------#
for k in weights_quant.keys():

    WEIGHT[ORDER[k]] = torch.round(WEIGHT[ORDER[k]] / SCALE_TABLE[k])

    try:
        WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]][:, :, [2, 1, 0], :]

    WEIGHT[ORDER[k]] = list(WEIGHT[ORDER[k]].flatten().numpy().astype(int))

    if k in get_bias_list(opt.PATH_CSV):
        ANS += struct.pack(str(len(WEIGHT[ORDER[k]])) + 'i', *WEIGHT[ORDER[k]])
    else:
        ANS += struct.pack(str(len(WEIGHT[ORDER[k]])) + 'h', *WEIGHT[ORDER[k]])


with open('json&raw/YoloV5_quantized_INQ.raw', 'wb+') as f:
    f.write(ANS)
    f.close()
