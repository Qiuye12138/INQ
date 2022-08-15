import torch
import argparse
from get_the_order import ORDER
from utils import bin2numpy_fp32



#-------------------------------------#
#       文件路径
#-------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--PATH_MODEL', type = str, default = 'weights/base.torchscript')
parser.add_argument('--PATH_FLOAT', type = str, default = 'json&raw/YoloV5_normed.raw')
opt = parser.parse_args()



#-------------------------------------#
#       解析权重
#-------------------------------------#
weights_float = bin2numpy_fp32(opt.PATH_FLOAT)
WEIGHT = torch.load(opt.PATH_MODEL).state_dict()



#-------------------------------------#
#       转换
#-------------------------------------#
for k in weights_float.keys():

    raw_float_all = torch.tensor(weights_float[k])

    try:
        raw_float_all = raw_float_all.reshape(WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous().shape).contiguous()
        raw_float_all = raw_float_all.permute(3, 2, 0, 1).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        raw_float_all = raw_float_all[:, [2, 1, 0], :, :]

    WEIGHT[ORDER[k]] = raw_float_all


torch.save(WEIGHT, 'weights/norm.pth')
