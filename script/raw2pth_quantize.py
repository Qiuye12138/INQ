import torch
import csv
import struct
from tqdm import tqdm
from get_the_order import ORDER, QWERT



#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/base.torchscript'
PATH_QUANT = 'json&raw/YoloV5_quantized.raw'                     # 定点raw路径
PATH_CSV   = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'        # csv路径



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
raw_addr_quant = {}
raw_size_quant = {}
weights_quant  = {}

raw_id   = []

whole_length_quant = len(open(PATH_QUANT, 'rb').read())

binfile_quant = open(PATH_QUANT, 'rb')

layer_num = struct.unpack('<i', binfile_quant.read(4))[0]

for i in range(layer_num):
    layer_id   = struct.unpack('<i', binfile_quant.read(4))[0]
    start_addr_quant = struct.unpack('<i', binfile_quant.read(4))[0]
    raw_addr_quant[layer_id] = start_addr_quant
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


model = torch.load(PATH_MODEL)
WEIGHT = model.state_dict()



#-------------------------------------#
#       转换
#-------------------------------------#
for k in tqdm(weights_quant.keys()):

    raw_fixed_all = torch.tensor(weights_quant[k]) * dic[k][1]                        # 定点权重

    try:
        raw_fixed_all = raw_fixed_all.reshape(WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous().shape).contiguous()
        raw_fixed_all = raw_fixed_all.permute(3, 2, 0, 1).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        raw_fixed_all = raw_fixed_all[:,[2,1,0],:,:]

    WEIGHT[ORDER[k]] = raw_fixed_all


torch.save(WEIGHT, 'weights/quantize.pth')
