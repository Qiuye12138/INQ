```bash
目录结构
├── assets
│   ├── val2017                     # 自行下载
│   └── instances_val2017.json      # 自行下载
├── configs
│   ├── YoloV5_8bit.py
│   └── YoloV5_16bit.ini
├── images
│   └── pic_list_map.txt
├── json&raw                        # 由icraft compile生成
├── logs                            # 由icraft compile生成
├── names
│   └── coco.names
├── res                             # 由python script\val.py生成
├── script
│   ├── create_ICRAFT.py
│   ├── get_the_order.py
│   ├── ICRAFT.py                   # 由python script\create_ICRAFT.py生成
│   ├── pth2raw.py
│   ├── raw2pth.py
│   ├── utils.py
│   └── val.py
└── weights
    ├── base.torchscript            # 基准模型，由第一章生成
    ├── INQ99.pth                   # 最终模型，由第三章生成
    ├── norm.pth                    # 由python script\raw2pth_norm.py生成
    └── quantize.pth                # 由python script\raw2pth_quantize.py生成
```

# 一、得到基准模型

> 第一章所有命令均在`ICraft_yolov5`仓库中运行

1.1、克隆工程[ICraft_yolov5: Fine tuning YOLOv5 for Icraft](https://github.com/Qiuye12138/ICraft_yolov5)，当前处于`master`分支

1.2、该仓库提供了一个预训练模型`yolov5n.pt`，测试其浮点精度（其他自行训练的模型也如此）

```bash
python3 val.py --data coco.yaml --weights yolov5n.pt --iou-thres 0.65
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.280
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.457
```

1.3、导出模型，这会将模型中的`BatchNorm`层融合

```bash
# 将在yolov5n.pt附近生成yolov5n.torchscript
python3 export.py --weights yolov5n.pt --include torchscript 
```

1.4、切换到`Icraft`分支上

```bash
# 该分支在创建模型时去除了模型中的BatchNorm层，与正常模型不通用
git checkout Icraft
```

1.5、`fuse`重训练

> 该训练的目的：
>
> - 得到一个原生不带`BatchNorm`的模型供后续实验使用（受原`YoloV5`工程限制）
> - 确认模型已经收敛，继续训练确实无法提升精度（防止重训练本身提升精度干扰`INQ`）
>
> `--weights` : 强制重新从`yaml`创建模型，而不是加载已有模型
>
> `train_fuse.py`创建新模型后，使用`yolov5n.torchscript`内的参数初始化权重

```bash
python3 train_fuse.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --batch-size 128 --device '1' --epochs 30 --name fuse
```

1.6、测试新模型精度

```bash
python3 val.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --iou-thres 0.65
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.280
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.457
```

1.7、导出模型

```bash
# 将在best.pt附近生成best.torchscript
python3 export.py --weights runs/train/fuse/weights/best.pt --include torchscript 

# 重命名为base.torchscript，作为后续实验的基准
mv runs/train/fuse/weights/best.torchscript runs/train/fuse/weights/base.torchscript
```



# 二、Icraft编译

> 第二章所有命令均在当前仓库中运行

2.1、将模型`base.torchscript`复制到`weights`文件夹下

2.2、使用`Icraft`编译模型

```bash
# 8比特
icraft compile configs\YoloV5_8bit.ini

# 16比特
icraft compile configs\YoloV5_16bit.ini
```

2.3、`Icraft.raw`转回`Pytorch.pth`

```bash
# 生成norm.pth
python script\raw2pth.py --PATH_MODEL 'weights/base.torchscript' --PATH_RAW 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv' --bit 32

# 生成quantize.pth--8比特
python script\raw2pth.py --PATH_MODEL 'weights/base.torchscript' --PATH_RAW 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv' --bit 8

# 生成quantize.pth--16比特
python script\raw2pth.py --PATH_MODEL 'weights/base.torchscript' --PATH_RAW 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv' --bit 16
```

2.4、生成`Icraft`参数

```bash
# 生成ICRAFT.py
python script\create_ICRAFT.py
```



# 三、INQ

> 第三章所有命令均在`ICraft_yolov5`仓库中运行

3.1、切换到`INQ`分支

```bash
# 该分支模拟了Icraft将特征图归一化后的运算，与正常模型不通用
# 8比特
git checkout INQ

# 16比特
git checkout INQ_16bit
```

3.2、将`ICRAFT.py`复制到当前路径下；将`norm.pth`、`quantize.pth`复制到`runs/train/fuse/weights/`

3.3、测试`norm`精度

> 事实上由于添加了特征图量化算子，该步骤已经无法复现
>
> 除非你手动把`models/common.py`内`FakeQuantize`的`forward`改为`return x`，忽略量化
>
> 记得改回来

```bash
python3 val_norm.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --iou-thres 0.65
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.280
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.457
```

3.4、测试`quantize`精度

```bash
# 8比特
python3 val_quantize.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --iou-thres 0.65
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.244  ↓3.6%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.419  ↓3.8%

# 16比特
python3 val_quantize.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --iou-thres 0.65
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.257  ↓2.3%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.430  ↓2.7%
```

3.5、开始`INQ`

```bash
auto.sh
```

3.6、测试`INQ`精度

```bash
# 8比特
python3 val.py --data coco.yaml --weights runs/train/INQ99/weights/best.pt --iou-thres 0.65
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.263  ↓1.7%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.442  ↓1.5%

# 16比特
python3 val.py --data coco.yaml --weights runs/train/INQ99/weights/best.pt --iou-thres 0.65
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.  ↓%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.  ↓%
```

3.7、`Pytorch.pth`转回`Icraft.raw`

```bash
# 将在json&raw文件夹下生成YoloV5_quantized_INQ.raw
# 8比特
python script\pth2raw_quantize.py --PATH_MODEL 'weights/INQ99.pth' --PATH_QUANT 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'  --bit 8

# 16比特
python script\pth2raw_quantize.py --PATH_MODEL 'weights/INQ99.pth' --PATH_QUANT 'json&raw/YoloV5_quantized.raw' --PATH_CSV 'logs/quantizer/BUYI/YoloV5/YoloV5_raws.csv'  --bit 16


# 复制一份json
cp json&raw\YoloV5_quantized.json json&raw\YoloV5_quantized_INQ.json

# 求md5
certutil -hashfile json&raw\YoloV5_quantized_INQ.raw MD5    # Windows
md5sum json&raw/YoloV5_quantized_INQ.raw MD5                # Linux

# 将json&raw\YoloV5_quantized_INQ.json内的"raw_md5"改为YoloV5_quantized_INQ.raw的md5值
```

3.8、效果测试

```bash
# Icraft浮点
python script\val.py --JSON_PATH 'json&raw/YoloV5_optimized.json' --RAW_PATH 'json&raw/YoloV5_optimized.raw' --RES_PATH 'res/FLOAT/'
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.260
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.434

########################################################################################################################

# Icraft定点-INQ前--8bit
python script\val.py --JSON_PATH 'json&raw/YoloV5_quantized.json' --RAW_PATH 'json&raw/YoloV5_quantized.raw' --RES_PATH 'res/QUANT/' --QUANT
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.226  ↓3.4%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.394  ↓4.0%

# Icraft定点-INQ后--8bit
python script\val.py --JSON_PATH 'json&raw/YoloV5_quantized_INQ.json' --RAW_PATH 'json&raw/YoloV5_quantized_INQ.raw' --RES_PATH 'res/QUANTINQ/' --QUANT
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.233  ↓2.7%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.407  ↓2.7%

########################################################################################################################

# Icraft定点-INQ前--16bit
python script\val.py --JSON_PATH 'json&raw/YoloV5_quantized.json' --RAW_PATH 'json&raw/YoloV5_quantized.raw' --RES_PATH 'res/QUANT/' --QUANT
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.245  ↓1.5%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.413  ↓2.1%

# Icraft定点-INQ后--16bit
python script\val.py --JSON_PATH 'json&raw/YoloV5_quantized_INQ.json' --RAW_PATH 'json&raw/YoloV5_quantized_INQ.raw' --RES_PATH 'res/QUANTINQ/' --QUANT
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.242  ↓1.8%
# Average Precision (AP) @[ IoU=0.50      | area= all | maxDets=100 ] = 0.411  ↓2.3%
```

