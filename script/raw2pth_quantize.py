import torch
import argparse
from get_the_order import ORDER
from utils import bin2numpy_int8, bin2numpy_int16, get_scale_dict



#-------------------------------------#
#       文件路径
#-------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--PATH_MODEL', type = str, default = 'weights/base.torchscript')
parser.add_argument('--PATH_QUANT', type = str, default = 'json&raw/YoloV5_quantized.raw')
parser.add_argument('--PATH_CSV'  , type = str, default = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv')
parser.add_argument('--bit'       , type = int, default = 8, choices=[8, 16])
opt = parser.parse_args()



#-------------------------------------#
#       解析Scale
#-------------------------------------#
SCALE_TABLE = get_scale_dict(opt.PATH_CSV)



#-------------------------------------#
#       解析权重
#-------------------------------------#
if opt.bit == 8:
    weights_quant = bin2numpy_int8(opt.PATH_QUANT, opt.PATH_CSV)
else:
    weights_quant = bin2numpy_int16(opt.PATH_QUANT, opt.PATH_CSV)

WEIGHT = torch.load(opt.PATH_MODEL).state_dict()



#-------------------------------------#
#       转换
#-------------------------------------#
for k in weights_quant.keys():

    raw_fixed_all = torch.tensor(weights_quant[k]) * SCALE_TABLE[k]

    try:
        raw_fixed_all = raw_fixed_all.reshape(WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous().shape).contiguous()
        raw_fixed_all = raw_fixed_all.permute(3, 2, 0, 1).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        raw_fixed_all = raw_fixed_all[:, [2, 1, 0], :, :]

    WEIGHT[ORDER[k]] = raw_fixed_all


torch.save(WEIGHT, 'weights/quantize.pth')
