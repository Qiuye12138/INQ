import cv2
import math
import torch
import torchvision
import icraftBY.simulator as sim



#-------------------------------------#
#       文件路径
#-------------------------------------#
IMG_PATH  = 'assets/val2017/000000002006.jpg'
NAME_PATH = 'names/coco.names'
JSON_PATH = 'json&raw/YoloV5_parsed.json'
RAW_PATH  = 'json&raw/YoloV5_parsed.raw'

with open(NAME_PATH, 'r') as f:
    names = f.read().split('\n')



#-------------------------------------#
#       参数
#-------------------------------------#
CONF   = 0.5
NMS    = 0.65
NET_W  = 640
NET_H  = 640
STRIDE = [   8,  16,  32]
A_W    = [[ 10,  16,  33], [ 30,  62,   59], [116, 156, 373]]
A_H    = [[ 13,  30,  23], [ 61,  45,  119], [ 90, 198, 326]]



#-----------------------------------------#
#       前处理
#-----------------------------------------#
image  = cv2.imread(IMG_PATH)
IMG_H, IMG_W = image.shape[:2]
ratio = min(NET_W / IMG_W, NET_H / IMG_H)



#-------------------------------------#
#       加载模型
#-------------------------------------#
option_parser = {'json'     : JSON_PATH,
                 'raw'      : RAW_PATH,
                 'target'   : 'BUYI',
                 'fake_qf'  : 'false',
                 'cudamode' : 'false',
                 'show'     : 'false'}

network_quantized = sim.Network(option_parser)



#-------------------------------------#
#       推理
#-------------------------------------#
output_quantized = network_quantized.run('assets/val640/' + IMG_PATH.split('/')[-1])
a = torch.from_numpy(output_quantized[0]).permute(0, 3, 1, 2)
b = torch.from_numpy(output_quantized[1]).permute(0, 3, 1, 2)
c = torch.from_numpy(output_quantized[2]).permute(0, 3, 1, 2)

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

                    x1 = B_X - B_W / 2
                    y1 = B_Y - B_H / 2
                    x2 = B_X + B_W / 2
                    y2 = B_Y + B_H / 2

                    mscore, mindex = torch.max(BOX[5:], 0)

                    if (score / (1 + math.exp(-mscore.item())) > CONF):
                        scores.append(score / (1 + math.exp(-mscore.item())))
                        ccclas.append(mindex.item())
                        anchorBoxes.append(x1)
                        anchorBoxes.append(y1)
                        anchorBoxes.append(x2)
                        anchorBoxes.append(y2)


scores = torch.tensor(scores)
ccclas = torch.tensor(ccclas)
anchorBoxes = torch.tensor(anchorBoxes)
anchorBoxes = anchorBoxes.contiguous().view(-1, 4)

anchors_nms_idx = torchvision.ops.nms(anchorBoxes, scores, NMS)     # NMS，得到NMS后的真实框的id

for idx in anchors_nms_idx:

    x1 = int(anchorBoxes[idx][0].item())
    y1 = int(anchorBoxes[idx][1].item())
    x2 = int(anchorBoxes[idx][2].item())
    y2 = int(anchorBoxes[idx][3].item())

    caption = '{} {:.3f}'.format(names[int(ccclas[idx].item())], scores[idx].item()*100)

    cv2.putText(image, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

cv2.imshow('detections', image)
cv2.waitKey(0)
