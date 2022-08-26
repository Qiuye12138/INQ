```bash
目录结构
├── assets
│   ├── val2017                     # 自行下载
│   └── instances_val2017.json      # 自行下载
├── configs
│   ├── YoloV5_8bit.ini
│   └── YoloV5_16bit.ini
├── images
│   └── pic_list_map.txt
├── json&raw                        # 由icraft compile生成
├── logs                            # 由icraft compile生成
├── names
│   └── coco.names
├── script
│   ├── create_ICRAFT.py
│   ├── detect.py
│   ├── get_the_order.py
│   ├── ICRAFT.py                   # 由python script\create_ICRAFT.py生成
│   ├── plot.py
│   ├── pth2raw.py
│   ├── raw2pth.py
│   ├── utils.py
│   └── val.py
└── weights
    ├── base.torchscript            # 基准模型，由第一章生成
    ├── INQ99.torchscript           # 最终模型，由第三章生成
    ├── norm.pth                    # 由python script\raw2pth.py生成
    └── quantize.pth                # 由python script\raw2pth.py生成
```

# 一、得到基准模型

> 第一章所有命令均在`ICraft_yolov5`仓库中运行

## 1.1、克隆工程`ICraft_yolov5`

```bash
# 当前处于master分支
git clone https://github.com/Qiuye12138/ICraft_yolov5.git
```

## 1.2、得到融合后权重

该仓库提供了两个预训练模型`yolov5n.pt`和`yolov5s.pt`，以下将以`YoloV5n`为例

```bash
# 将在yolov5n.pt附近生成yolov5n.torchscript
python3 export.py --weights yolov5n.pt --include torchscript
```

## 1.3、`fuse`重训练

> 该训练的目的：
>
> - 得到一个原生不带`BatchNorm`的模型供后续实验使用（受原`YoloV5`工程限制）
> - 确认模型已经收敛，继续训练确实无法提升精度（防止重训练本身提升精度干扰`INQ`）
>
> `--weights` : 强制重新从`yaml`创建模型，而不是加载已有模型
>
> `train.py`创建新模型后，使用`yolov5*.torchscript`内的参数初始化权重

```bash
# 该分支在创建模型时去除了模型中的BatchNorm层，与正常模型不通用
git checkout Icraft

python3 train.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --preweight yolov5n.torchscript --batch-size 128 --name fuse --patience 10
```

## 1.4、导出模型

```bash
# 将在best.pt附近生成best.torchscript
python3 export.py --weights runs/train/fuse/weights/best.pt --include torchscript

# 重命名为base.torchscript，作为后续实验的基准
mv runs/train/fuse/weights/best.torchscript runs/train/fuse/weights/base.torchscript
```



# 二、Icraft编译

> 第二章所有命令均在当前仓库中运行

## 2.1、使用`Icraft`编译模型

```bash
# 务必先将模型base.torchscript复制到weights文件夹下
# 8比特
icraft compile configs\YoloV5_8bit.ini

# 16比特
icraft compile configs\YoloV5_16bit.ini
```


## 2.2、`Icraft.raw`转回`Pytorch.pth`

```bash
# 生成norm.pth
python script\raw2pth.py --PATH_MODEL 'weights/base.torchscript' --PATH_RAW 'json&raw/YoloV5_normed.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv' --bit 32

# 生成quantize.pth--8比特
python script\raw2pth.py --PATH_MODEL 'weights/base.torchscript' --PATH_RAW 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv' --bit 8

# 生成quantize.pth--16比特
python script\raw2pth.py --PATH_MODEL 'weights/base.torchscript' --PATH_RAW 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv' --bit 16
```

## 2.3、生成`Icraft`参数

```bash
# 生成ICRAFT.py
python script\create_ICRAFT.py
```



# 三、INQ

> 第三章所有命令均在`ICraft_yolov5`仓库中运行

## 3.1、代码准备

需要对`Pytorch`做一些修改，才能使用`INQ`

- 为`Tensor`类添加属性`MaskMatrix`，作为逐元素冻结权重所需要的掩膜

  ```bash
  # 我的路径 ： /usr/local/lib/python3.6/dist-packages/torch/_tensor.py
  MaskMatrix = None
  ```

- 在优化器更新梯度前，将梯度与掩膜相乘

  ```bash
  # 我的路径 ： /usr/local/lib/python3.6/dist-packages/torch/optim/_functional.py
  if param.MaskMatrix != None:
      param.add_(d_p*param.MaskMatrix, alpha=-lr)
  ```

## 3.2、文件准备

将`ICRAFT.py`复制到当前路径下；将`norm.pth`、`quantize.pth`复制到`runs/train/fuse/weights/`

## 3.3、开始`INQ`

```bash
# 该分支模拟了Icraft将特征图归一化后的运算，与正常模型不通用
git checkout INQ

python3 train.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --preweight runs/train/fuse/weights/quantize.pth --batch-size 128 --name INQ50  --patience 10 --ratio 0.5
python3 train.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --preweight runs/train/INQ50/weights/best.pt     --batch-size 128 --name INQ75  --patience 10 --ratio 0.75
python3 train.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --preweight runs/train/INQ75/weights/best.pt     --batch-size 128 --name INQ875 --patience 10 --ratio 0.875
python3 train.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --preweight runs/train/INQ875/weights/best.pt    --batch-size 128 --name INQ95  --patience 10 --ratio 0.75
python3 train.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --preweight runs/train/INQ95/weights/best.pt     --batch-size 128 --name INQ99  --patience 10 --ratio 0.99
```

## 3.4、`Pytorch.pth`转回`Icraft.raw`

```bash
# 将在json&raw文件夹下生成YoloV5_quantized_INQ.raw
# 8比特
python script\pth2raw.py --PATH_MODEL 'weights/INQ99.torchscript' --PATH_QUANT 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'  --bit 8

# 16比特
python script\pth2raw.py --PATH_MODEL 'weights/INQ99.torchscript' --PATH_QUANT 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'  --bit 16


# 复制一份json
cp 'json&raw\YoloV5_quantized.json' 'json&raw\YoloV5_quantized_INQ.json'

# 求md5
certutil -hashfile 'json&raw\YoloV5_quantized_INQ.raw' MD5  # Windows
md5sum json&raw/YoloV5_quantized_INQ.raw MD5                # Linux

# 将json&raw\YoloV5_quantized_INQ.json内的"raw_md5"改为YoloV5_quantized_INQ.raw的md5值
```



# 四、小工具

## 4.1、画图

```bash
# 看量化前后权重分布
python script\plot.py --PATH_MODEL0 'weights/norm.pth' --PATH_MODEL1 'weights/quantize.pth'

# 看INQ前后权重分布
python script\plot.py --PATH_MODEL0 'weights/quantize.pth' --PATH_MODEL1 'weights/INQ99.torchscript' --ratio 0.5
```

## 4.2、保存每层特征图

```bash
# 浮点特征图
python script\detect.py --JSON_PATH 'json&raw/YoloV5_optimized.json' --RAW_PATH 'json&raw/YoloV5_optimized.raw'

# 定点特征图
python script\detect.py --JSON_PATH 'json&raw/YoloV5_quantized.json' --RAW_PATH 'json&raw/YoloV5_quantized.raw' --QUANT
```

## 4.3、测试精度

```bash
# Icraft浮点
python script\val.py --JSON_PATH 'json&raw/YoloV5_optimized.json' --RAW_PATH 'json&raw/YoloV5_optimized.raw'

# Icraft定点-INQ前
python script\val.py --JSON_PATH 'json&raw/YoloV5_quantized.json' --RAW_PATH 'json&raw/YoloV5_quantized.raw' --QUANT

# Icraft定点-INQ后
python script\val.py --JSON_PATH 'json&raw/YoloV5_quantized_INQ.json' --RAW_PATH 'json&raw/YoloV5_quantized_INQ.raw' --QUANT
```
