import csv
import struct



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
    若第三列max为空，则视为bias层，返回其第一列raw_id
    '''
    SCALE_TABLE = {}
    with open(PATH_CSV, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for row in f_csv:
            SCALE_TABLE[int(row[0])] = float(row[4])

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
