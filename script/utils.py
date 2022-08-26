import cv2
import csv
import torch
import struct
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



def get_bias_list(PATH_CSV):
    '''
    只接受raws.csv，不接受ftmps.csv
    若第三列max为空，则视为bias层，返回其第一列raw_id
    '''
    with open(PATH_CSV, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        return [int(row[0]) for row in f_csv if row[3] == '']


def get_scale_dict(PATH_CSV):
    '''
    只接受raws.csv，不接受ftmps.csv
    若其第五列scale
    '''
    SCALE_TABLE = {}
    with open(PATH_CSV, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            SCALE_TABLE[int(row[0])] = float(row[4].split(' ')[0])

    return SCALE_TABLE


def bin2numpy_fp32(PATH_RAW):
    '''
    只接受norm及之前的raw，不接受quantize及之后的raw
    返回一个字典，键为raw_id，值为一维numpy数组
    '''
    raw_addr_float = {}
    raw_size_float = {}
    weights_float  = {}

    raw_id   = []
    whole_length_float = len(open(PATH_RAW, 'rb').read())
    binfile_float = open(PATH_RAW, 'rb')
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

    return weights_float


def bin2numpy_int8(PATH_RAW, PATH_CSV):
    '''
    只接受quantize及之后的raw，不接受norm及之前的raw
    返回一个字典，键为raw_id，值为一维numpy数组
    '''
    raw_addr_quant = {}
    raw_size_quant = {}
    weights_quant  = {}

    raw_id   = []

    whole_length_quant = len(open(PATH_RAW, 'rb').read()) # 全部字节数

    binfile_quant = open(PATH_RAW, 'rb')

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

    return weights_quant


def bin2numpy_int16(PATH_RAW, PATH_CSV):
    '''
    只接受quantize及之后的raw，不接受norm及之前的raw
    返回一个字典，键为raw_id，值为一维numpy数组
    '''
    raw_addr_quant = {}
    raw_size_quant = {}
    weights_quant  = {}

    raw_id   = []

    whole_length_quant = len(open(PATH_RAW, 'rb').read()) # 全部字节数

    binfile_quant = open(PATH_RAW, 'rb')

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
            for j in range(0, raw_size_quant[i], 2):
                data = struct.unpack('<h', binfile_quant.read(2))[0]
                weights_quant[i].append(data)
        else:
            for j in range(0, raw_size_quant[i], 4):
                data = struct.unpack('<i', binfile_quant.read(4))[0]
                weights_quant[i].append(data)

    return weights_quant


def letterbox(im, new_shape):

    shape = im.shape[:2]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return im, r, (dw, dh)


def scale_coords(coords, img0_shape, gain, pad):

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)

    return coords


def clip_coords(boxes, shape):

    boxes[:, 0].clamp_(0, shape[1])
    boxes[:, 1].clamp_(0, shape[0])
    boxes[:, 2].clamp_(0, shape[1])
    boxes[:, 3].clamp_(0, shape[0])


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):

    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def xywh2xyxy(x):

    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2

    return y


def non_max_suppression(prediction, conf_thres, iou_thres, multi_label=False):

    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    multi_label &= nc > 1

    output = [torch.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        if i.shape[0] > 300:
            i = i[:300]

        output[xi] = x[i]

    return output


def xyxy2xywh(x):

    y = x.clone()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]

    return y


def save_one_json(predn, jdict, image_id, class_map):

    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def make_grid(nx, ny, ANCHORS):

    shape = 1, 3, ny, nx, 2
    y, x = torch.arange(ny), torch.arange(nx)
    yv, xv = torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
    anchor_grid = (ANCHORS).view((1, 3, 1, 1, 2)).expand(shape)

    return grid, anchor_grid


def get_mAP(jdict):

    cocoGt = COCO('assets/instances_val2017.json')
    cocoDt = cocoGt.loadRes(jdict)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def smart_load(path):

    if path.endswith(('.torchscript', '.pt')) :
        weights = torch.load(path, map_location = torch.device('cpu')).state_dict()
    elif path.endswith('.pth'):
        weights = torch.load(path, map_location = torch.device('cpu'))

    return weights
