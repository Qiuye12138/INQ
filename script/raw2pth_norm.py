import torch
import struct
from tqdm import tqdm
import matplotlib.pyplot as plt
from get_the_order import ORDER, QWERT



#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/base.torchscript'
PATH_FLOAT = 'json&raw/YoloV5_normed.raw'                        # 浮点raw路径



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


model = torch.load(PATH_MODEL)
WEIGHT = model.state_dict()



#-------------------------------------#
#       转换
#-------------------------------------#
for k in tqdm(weights_float.keys()):

    raw_float_all = torch.tensor(weights_float[k])                                # 浮点权重

    try:
        raw_float_all = raw_float_all.reshape(WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous().shape).contiguous()
        raw_float_all = raw_float_all.permute(3, 2, 0, 1).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        raw_float_all = raw_float_all[:,[2,1,0],:,:]

    aa = raw_float_all.flatten().numpy()[0] / WEIGHT[ORDER[k]].flatten().numpy()[0]

    plt.plot(WEIGHT[ORDER[k]].flatten().numpy(), '.', color = 'red')
    plt.plot(raw_float_all.flatten().numpy() / aa, '.', color = 'green')

    plt.show()
    # plt.savefig(ORDER[k] + '.jpg')
    plt.cla()               # 清空画布

    WEIGHT[ORDER[k]] = raw_float_all


# torch.save(WEIGHT, 'weights/norm.pth')
