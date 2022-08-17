import os
import cv2
import json
import torch
import argparse
from tqdm import tqdm
import icraftBY.simulator as sim
from utils import make_grid, non_max_suppression, scale_coords, save_one_json, letterbox, get_mAP



#-------------------------------------#
#       文件路径
#-------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--JSON_PATH', type = str, default = 'json&raw/YoloV5_optimized.json')
parser.add_argument('--RAW_PATH' , type = str, default = 'json&raw/YoloV5_optimized.raw')
parser.add_argument('--QUANT'    , const = True, nargs = '?', default = False)

opt = parser.parse_args()



#-------------------------------------#
#       获取SCALE
#-------------------------------------#
if opt.QUANT:
    SCALE = []
    with open(opt.JSON_PATH, 'r') as f:
        file = json.load(f)
        for i in range(3):
            SCALE.append(file['operations'][-1]['input_ftmp'][i]['norm_ratio'][0]['value'])



#-------------------------------------#
#       参数
#-------------------------------------#
CONF   = 0.001
NMS    = 0.65
NET_W  = 640
NET_H  = 640
STRIDE = [   8,  16,  32]
ANCHORS = torch.tensor([[[  10,  13], [  16,   30], [ 33,  23]],
                        [[  30,  61], [  62,   45], [ 59, 119]],
                        [[ 116,  90], [ 156,  198], [373, 326]]])

category_ids = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15,
16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
85, 86, 87, 88, 89, 90 ]



#-------------------------------------#
#       加载模型
#-------------------------------------#
option_parser = {'json'     : opt.JSON_PATH,
                 'raw'      : opt.RAW_PATH,
                 'target'   : 'BUYI',
                 'fake_qf'  : 'false',
                 'cudamode' : 'true',
                 'show'     : 'false'}

network_quantized = sim.Network(option_parser)


jdict = []
#-------------------------------------#
#       遍历测试集
#-------------------------------------#
for file in tqdm(os.listdir('assets/val2017/')):
    #-----------------------------------------#
    #       前处理
    #-----------------------------------------#
    image  = cv2.imread('assets/val2017/' + file)
    IMG_H, IMG_W = image.shape[:2]
    resized_img, ratio, (dw, dh) = letterbox(image, (NET_H, NET_W))
    resized_img = resized_img[None]



    #-------------------------------------#
    #       推理
    #-------------------------------------#
    output_quantized = network_quantized.run([resized_img])
    a = torch.from_numpy(output_quantized[0]).permute(0, 3, 1, 2)
    b = torch.from_numpy(output_quantized[1]).permute(0, 3, 1, 2)
    c = torch.from_numpy(output_quantized[2]).permute(0, 3, 1, 2)

    if opt.QUANT:
        a *= SCALE[0]
        b *= SCALE[1]
        c *= SCALE[2]

    T = [a, b, c]



    #-----------------------------------------#
    #       后处理
    #-----------------------------------------#
    z = []
    for i in range(3):
        bs, _, ny, nx = T[i].shape
        T[i] = T[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = T[i].sigmoid()
        grid, anchor_grid = make_grid(nx, ny, ANCHORS[i])
        y[..., 0:2] = (y[..., 0:2] * 2 + grid) * STRIDE[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid
        z.append(y.view(bs, -1, 85))

    Z = torch.cat(z, 1)
    out = non_max_suppression(Z, CONF, NMS, multi_label = True)

    for si, pred in enumerate(out):
        predn = pred.clone()
        scale_coords(predn[:, :4], (IMG_H, IMG_W), ratio, (dw, dh))
        save_one_json(predn, jdict, int(file.split('.')[0]), category_ids)


get_mAP(jdict)
