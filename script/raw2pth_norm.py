import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from get_the_order import ORDER
from utils import bin2numpy_fp32


#-------------------------------------#
#       文件路径
#-------------------------------------#
PATH_MODEL = 'weights/base.torchscript'
PATH_FLOAT = 'json&raw/YoloV5_normed.raw'



#-------------------------------------#
#       解析权重
#-------------------------------------#
weights_float = bin2numpy_fp32(PATH_FLOAT)
WEIGHT = torch.load(PATH_MODEL).state_dict()



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


torch.save(WEIGHT, 'weights/norm.pth')
