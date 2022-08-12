import torch
import struct
from tqdm import tqdm
from get_the_order import ORDER
from utils import get_bias_list, get_scale_dict



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

    if i not in get_bias_list(PATH_CSV):
        for j in range(0, raw_size_quant[i], 1):
            data = struct.unpack('<b', binfile_quant.read(1))[0]
            weights_quant[i].append(data)
    else:
        for j in range(0, raw_size_quant[i], 2):
            data = struct.unpack('<h', binfile_quant.read(2))[0]
            weights_quant[i].append(data)


WEIGHT = torch.load(PATH_MODEL, map_location = torch.device('cpu'))


#-------------------------------------#
#       转换
#-------------------------------------#
for k in tqdm(weights_quant.keys()):

    WEIGHT[ORDER[k]] = torch.round(WEIGHT[ORDER[k]] / SCALE_TABLE[k])

    try:
        WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]][:, :, [2, 1, 0], :]

    WEIGHT[ORDER[k]] = WEIGHT[ORDER[k]].flatten().numpy()

    x = WEIGHT[ORDER[k]]
    y = weights_quant[k]

    if k not in get_bias_list(PATH_CSV):
        for nnn in WEIGHT[ORDER[k]]:
            ANS += struct.pack('b', int(nnn))
    else:
        for nnn in WEIGHT[ORDER[k]]:
            ANS += struct.pack('h', int(nnn))


with open('json&raw/YoloV5_quantized_INQ.raw', 'wb+') as f:
    f.write(ANS)
    f.close()
