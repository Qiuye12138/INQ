import torch
import struct
import numpy as np



#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/base.torchscript'
PATH_FLOAT = 'json&raw/YoloV5_parsed.raw'



#-------------------------------------#
#       解析权重
#-------------------------------------#
raw_addr_float = {}
raw_size_float = {}
weights_float  = {}

raw_id   = []

whole_length_float = len(open(PATH_FLOAT, 'rb').read())
binfile_float = open(PATH_FLOAT, 'rb')
layer_num = struct.unpack('<i', binfile_float.read(4))[0]


for i in range(layer_num):
    layer_id   = struct.unpack('<i', binfile_float.read(4))[0]
    start_addr_float = struct.unpack('<i', binfile_float.read(4))[0]
    raw_addr_float[layer_id] = start_addr_float
    raw_id.append(layer_id)


for i in range(len(raw_id)-1):
    raw_size_float[raw_id[i]] = raw_addr_float[raw_id[i+1]] - raw_addr_float[raw_id[i]]


raw_size_float[raw_id[i+1]] = whole_length_float - raw_addr_float[raw_id[i+1]]


for i in raw_id:
    weights_float[i] = []
    for j in range(0, raw_size_float[i], 4):
        data = struct.unpack('<f', binfile_float.read(4))[0]
        weights_float[i].append(data)


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
