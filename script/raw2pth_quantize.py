import torch
from tqdm import tqdm
from get_the_order import ORDER
from utils import bin2numpy_int8
from utils import get_scale_dict



#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/base.torchscript'
PATH_QUANT = 'json&raw/YoloV5_quantized.raw'                     # 定点raw路径
PATH_CSV   = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'        # csv路径



#-------------------------------------#
#       解析Scale
#-------------------------------------#
SCALE_TABLE = get_scale_dict(PATH_CSV)



#-------------------------------------#
#       解析权重
#-------------------------------------#
weights_quant = bin2numpy_int8(PATH_QUANT, PATH_CSV)
WEIGHT = torch.load(PATH_MODEL).state_dict()



#-------------------------------------#
#       转换
#-------------------------------------#
for k in tqdm(weights_quant.keys()):

    raw_fixed_all = torch.tensor(weights_quant[k]) * SCALE_TABLE[k]

    try:
        raw_fixed_all = raw_fixed_all.reshape(WEIGHT[ORDER[k]].permute(2, 3, 1, 0).contiguous().shape).contiguous()
        raw_fixed_all = raw_fixed_all.permute(3, 2, 0, 1).contiguous()
    except:
        ...

    if ORDER[k] == 'model.0.conv.weight':
        raw_fixed_all = raw_fixed_all[:,[2,1,0],:,:]

    WEIGHT[ORDER[k]] = raw_fixed_all


torch.save(WEIGHT, 'weights/quantize.pth')
