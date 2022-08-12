import torch
import numpy as np
from utils import bin2numpy_fp32


#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/base.torchscript'
PATH_FLOAT = 'json&raw/YoloV5_parsed.raw'



#-------------------------------------#
#       解析权重
#-------------------------------------#
weights_float = bin2numpy_fp32(PATH_FLOAT)
model = torch.load(PATH_MODEL).state_dict()


ORDER = {}
QWERT = {}
#-------------------------------------#
#       
#-------------------------------------#
for k in weights_float.keys():

    raw_float_all = np.array(weights_float[k])                                # 浮点权重

    for key in model.keys():
        if raw_float_all[0] == model[key].flatten().numpy()[0]:
            ORDER[k] = key
            QWERT[key] = k

# print(ORDER)
# print(QWERT)
