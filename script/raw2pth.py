import torch
import argparse
from get_the_order import ORDER
from utils import *



#-------------------------------------#
#       文件路径
#-------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--PATH_MODEL', type = str, default = 'weights/base.torchscript')
parser.add_argument('--PATH_RAW'  , type = str, default = 'json&raw/YoloV5_quantized.raw')
parser.add_argument('--PATH_CSV'  , type = str, default = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv')
parser.add_argument('--bit'       , type = int, default = 12, choices=[12, 32])
opt = parser.parse_args()



#-------------------------------------#
#       解析Scale
#-------------------------------------#
SCALE_TABLE = get_scale_dict(opt.PATH_CSV)



#-------------------------------------#
#       解析权重
#-------------------------------------#
if opt.bit == 12:
    weights = bin2numpy_int16(opt.PATH_RAW, opt.PATH_CSV)
else:
    weights = bin2numpy_fp32(opt.PATH_RAW)

WEIGHT = smart_load(opt.PATH_MODEL)



#-------------------------------------#
#       转换
#-------------------------------------#
for k in weights.keys():

    if opt.bit == 32:
        raw = torch.tensor(weights[k])
    else:
        raw = torch.tensor(weights[k]) * SCALE_TABLE[k]

    try:
        raw = raw.reshape(WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous().shape).contiguous()
        raw = raw.permute(3, 2, 0, 1).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        raw = raw[:, [2, 1, 0], :, :]

    WEIGHT[ORDER[k]] = raw

if opt.bit == 32:
    torch.save(WEIGHT, 'weights/norm.pth')
else:
    torch.save(WEIGHT, 'weights/quantize.pth')
