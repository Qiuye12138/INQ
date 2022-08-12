import os
import re
import json
from tqdm import tqdm


def yolores2cocores(originImagesDir, resultsDir, dtJsonPath):

    indexes = sorted(os.listdir(originImagesDir))
    dataset = []
    imgIds = []
    
    pattern = re.compile("^0+")

    ann_id_cnt = 0
    for index in tqdm(indexes):

        txtFile = index.replace('.jpg','.txt').replace('.png','.txt')
        img_id = int(re.sub(pattern, "", index).split('.')[0])

        if not os.path.exists(os.path.join(resultsDir, txtFile)):
            continue

        with open(os.path.join(resultsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x1 = float(label[1])
                y1 = float(label[2])
                width = float(label[3])
                height = float(label[4])
            
                cls_id = int(label[0]) 
                score = float(label[5])

                dataset.append({'image_id'   : img_id,
                                'category_id': cls_id,
                                'bbox'       : [x1, y1, width, height],
                                'score'      : score})
                ann_id_cnt += 1
                imgIds.append(img_id)

    with open(dtJsonPath, 'w') as f:
        json.dump(dataset, f)

    return imgIds