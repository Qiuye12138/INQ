import numpy as np
from utils import *


#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/base.torchscript'
PATH_FLOAT = 'json&raw/YoloV5_parsed.raw'



#-------------------------------------#
#       解析权重
#-------------------------------------#
weights_float = bin2numpy_fp32(PATH_FLOAT)
model = smart_load(PATH_MODEL)


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
