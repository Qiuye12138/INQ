import csv
import json
import argparse
from get_the_order import ORDER



#-------------------------------------#
#       参数
#-------------------------------------#
左 = [ 6, 20, 25, 39, 44, 49, 63]
右 = [10, 24, 29, 43, 48, 53, 67]
和 = [11, 25, 30, 44, 49, 54, 68]
输出层  = [140, 141, 142]
TABLE_F = {}
TABLE_R = {}
SILU_I  = []
SILU_O  = []



#-------------------------------------#
#       文件路径
#-------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--JSON_PATH', type = str, default = 'json&raw/YoloV5_quantized.json')
parser.add_argument('--FCSV_PATH', type = str, default = 'logs/quantizer/BUYI/YoloV5/YoloV5_ftmps.csv')
parser.add_argument('--RCSV_PATH'  , type = str, default = 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv')
opt = parser.parse_args()



#-------------------------------------#
#       解析normratio
#-------------------------------------#
with open(opt.FCSV_PATH, 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        try:
            TABLE_F[int(row[0])] = float(row[4].split(' ')[-1][:-1])
        except:
            continue



#-------------------------------------#
#       解析Scale
#-------------------------------------#
with open(opt.RCSV_PATH, 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        try:
            TABLE_R[int(row[0])] = 2 ** -float(row[6].split(' ')[0])
        except:
            continue



#-------------------------------------#
#       解析SILU
#-------------------------------------#
with open(opt.JSON_PATH, 'r') as f:
    JSON = json.load(f)
    for op in JSON['operations']:
        if op['op_type'] == 'Silu':
            SILU_I.append(op['input_ftmp'][0]['norm_ratio'][0]['value'])
            SILU_O.append(op['output_ftmp'][0]['norm_ratio'][0]['value'])



#-------------------------------------#
#       输出文件
#-------------------------------------#
with open('script/ICRAFT.py', 'w') as f:
    f.write('from itertools import cycle\n\n')

    f.write('SUM_L  = cycle([')
    for i in 左:
        f.write(str(TABLE_F[i]) + ', ')
    f.write('])\n')

    f.write('SUM_R  = cycle([')
    for i in 右:
        f.write(str(TABLE_F[i]) + ', ')
    f.write('])\n')

    f.write('SUM_A  = cycle([')
    for i in 和:
        f.write(str(TABLE_F[i]) + ', ')
    f.write('])\n')

    f.write('SILU_I = cycle([')
    for i in SILU_I:
        f.write(str(i) + ', ')
    f.write('])\n')

    f.write('SILU_O = cycle([')
    for i in SILU_O:
        f.write(str(i) + ', ')
    f.write('])\n')

    f.write('RATIO  = [')
    for i in 输出层:
        f.write(str(TABLE_F[i]) + ', ')
    f.write(']\n')


    f.write('SCALE_TABLE = {\n')
    for key in TABLE_R.keys():
        f.write("'" + ORDER[key] + "' : " + str(TABLE_R[key]) + ',\n')
    f.write('}\n')
