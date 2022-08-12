import torch
import struct
from get_the_order import ORDER
from utils import get_bias_list, get_scale_dict, bin2numpy_int8



#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/INQ99.pth'
PATH_QUANT = 'json&raw/YoloV5_quantized.raw'
PATH_CSV   = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'



#-------------------------------------#
#       解析Scale
#-------------------------------------#
SCALE_TABLE = get_scale_dict(PATH_CSV)



#-------------------------------------#
#       解析权重
#-------------------------------------#
binfile_quant = open(PATH_QUANT, 'rb')
layer_num = struct.unpack('<i', binfile_quant.read(4))[0]
ANS = struct.pack('i', layer_num)
ANS += binfile_quant.read(layer_num * 2 * 4)

weights_quant = bin2numpy_int8(PATH_QUANT, PATH_CSV)
WEIGHT = torch.load(PATH_MODEL, map_location = torch.device('cpu'))



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

    if k in get_bias_list(PATH_CSV):
        ANS += struct.pack(str(len(WEIGHT[ORDER[k]])) + 'h', *WEIGHT[ORDER[k]])
    else:
        ANS += struct.pack(str(len(WEIGHT[ORDER[k]])) + 'b', *WEIGHT[ORDER[k]])


with open('json&raw/YoloV5_quantized_INQ.raw', 'wb+') as f:
    f.write(ANS)
    f.close()
