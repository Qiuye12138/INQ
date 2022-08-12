import os
import cv2
import json
import math
import torch
import shutil
import torchvision
from tqdm import tqdm
from utils import yolores2cocores
import icraftBY.simulator as sim
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



#-------------------------------------#
#       文件路径
#-------------------------------------#
JSON_PATH = 'json&raw/YoloV5_quantized_INQ.json'
RAW_PATH  = 'json&raw/YoloV5_quantized_INQ.raw'
RES_PATH  = 'res/QUANT_INQ/'

shutil.rmtree(RES_PATH, ignore_errors = True)
os.makedirs(RES_PATH)



#-------------------------------------#
#       获取SCALE
#-------------------------------------#
SCALE = []
with open(JSON_PATH, 'r') as f:
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
A_W    = [[ 10,  16,  33], [ 30,  62,   59], [116, 156, 373]]
A_H    = [[ 13,  30,  23], [ 61,  45,  119], [ 90, 198, 326]]

category_ids = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15,
16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
85, 86, 87, 88, 89, 90 ]



#-------------------------------------#
#       加载模型
#-------------------------------------#
option_parser = {'json'     : JSON_PATH,
                 'raw'      : RAW_PATH,
                 'target'   : 'BUYI',
                 'fake_qf'  : 'false',
                 'cudamode' : 'true',
                 'show'     : 'false'}

network_quantized = sim.Network(option_parser)



#-------------------------------------#
#       遍历测试集
#-------------------------------------#
for file in tqdm(os.listdir('assets/val640/')):
    #-----------------------------------------#
    #       前处理
    #-----------------------------------------#
    img_path = 'assets/val2017/' + file
    image  = cv2.imread(img_path)
    IMG_H, IMG_W = image.shape[:2]
    ratio = min(NET_W / IMG_W, NET_H / IMG_H)



    #-------------------------------------#
    #       推理
    #-------------------------------------#
    output_quantized = network_quantized.run('assets/val640/' + file)
    a = torch.from_numpy(output_quantized[0] * SCALE[0]).permute(0, 3, 1, 2)
    b = torch.from_numpy(output_quantized[1] * SCALE[1]).permute(0, 3, 1, 2)
    c = torch.from_numpy(output_quantized[2] * SCALE[2]).permute(0, 3, 1, 2)

    T = [a, b, c]



    #-----------------------------------------#
    #       后处理
    #-----------------------------------------#
    anchorBoxes = []
    scores = []
    ccclas = []

    for i in range(3):
        N, C, H, W = T[i].shape
        for h in range(H):
            for w in range(W):
                for box in range(3):
                    BOX = T[i][0, box*85:(box+1)*85, h, w]

                    score = 1 / (1 + math.exp(-BOX[4].item()))

                    if (score > CONF):

                        T_X = 1 / (1 + math.exp(-BOX[0].item()))
                        T_Y = 1 / (1 + math.exp(-BOX[1].item()))
                        T_W = 1 / (1 + math.exp(-BOX[2].item()))
                        T_H = 1 / (1 + math.exp(-BOX[3].item()))

                        B_X = (2 * T_X - 0.5 + w) * STRIDE[i] / ratio
                        B_Y = (2 * T_Y - 0.5 + h) * STRIDE[i] / ratio

                        B_W = 4 * T_W * T_W * A_W[i][box] / ratio
                        B_H = 4 * T_H * T_H * A_H[i][box] / ratio

                        mscore, mindex = torch.max(BOX[5:], 0)

                        if (score / (1 + math.exp(-mscore.item())) > CONF):
                            scores.append(score / (1 + math.exp(-mscore.item())))
                            ccclas.append(mindex.item())
                            anchorBoxes.append(B_X - B_W / 2)
                            anchorBoxes.append(B_Y - B_H / 2)
                            anchorBoxes.append(B_X + B_W / 2)
                            anchorBoxes.append(B_Y + B_H / 2)


    scores = torch.tensor(scores)
    ccclas = torch.tensor(ccclas)
    anchorBoxes = torch.tensor(anchorBoxes)
    anchorBoxes = anchorBoxes.contiguous().view(-1, 4)

    anchors_nms_idx = torchvision.ops.nms(anchorBoxes, scores, NMS)     # NMS，得到NMS后的真实框的id


    with open(RES_PATH + file[:-4]+'.txt','w') as f:
        for idx in anchors_nms_idx:

            x1 = anchorBoxes[idx][0].item()
            y1 = anchorBoxes[idx][1].item()
            x2 = anchorBoxes[idx][2].item()
            y2 = anchorBoxes[idx][3].item()

            w = x2 - x1
            h = y2 - y1

            id_ = str(category_ids[int(ccclas[idx].item())]) + ' '
            x1 = str(int(x1)) + ' '
            y1 = str(int(y1)) + ' '
            w  = str(int(w)) + ' '
            h  = str(int(h)) + ' '
            s  = str(scores[idx].item())

            f.write(id_ + x1 + y1 + w + h + s + '\n')



JS_PATH = RES_PATH + 'yolov5s_predictions.json'

imgIds = yolores2cocores('assets/val2017/', RES_PATH, JS_PATH)
cocoGt=COCO('assets/instances_val2017.json')
cocoDt=cocoGt.loadRes(JS_PATH)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = sorted(imgIds)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
