```bash
目录结构
├── assets
│   ├── val2017                     # 自行下载
│   ├── val640                      # 由python script\getPadImg.py生成
│   └── instances_val2017.json      # 自行下载
├── configs
│   └── YoloV5_BY_V2.0.ini
├── images
│   └── pic_list_map.txt
├── json&raw                        # 由icraft compile生成
├── logs                            # 由icraft compile生成
├── names
│   └── coco.names
├── res                             # 由python script\val_xxx.py生成
├── script
│   ├── create_ICRAFT.py
│   ├── get_the_order.py
│   ├── getPadImg.py
│   ├── ICRAFT.py                   # 由python script\create_ICRAFT.py生成
│   ├── plot.py
│   ├── pth2raw_quantize.py
│   ├── raw2pth_norm.py
│   ├── raw2pth_quantize.py
│   ├── utils.py
│   ├── val_float.py
│   ├── val_quant_INQ.py
│   └── val_quant.py
└── weights
    ├── base.torchscript            # 基准模型，由第一章生成
    ├── INQ99.pth                   # 最终模型，由第三章生成
    ├── norm.pth                    # 由python script\raw2pth_norm.py生成
    └── quantize.pth                # 由python script\raw2pth_quantize.py生成
```

# 一、得到基准模型

1.1、克隆工程[ICraft_yolov5: Fine tuning YOLOv5 for Icraft](https://github.com/Qiuye12138/ICraft_yolov5)，当前处于`master`分支

> 第一章所有命令均在`ICraft_yolov5`仓库中运行

1.2、该仓库提供了一个预训练模型`yolov5n.pt`，测试其浮点精度（其他自行训练的模型也如此）

```bash
python3 val.py --data coco.yaml --weights yolov5n.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.457
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
> --weights : 强制重新从yaml创建模型，而不是加载已有模型

```bash
python3 train_fuse.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --batch-size 128 --device '1' --epochs 30 --name fuse
```

1.6、测试新模型精度

```bash
python3 val.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.457
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

2.1、生成测试图片

```bash
python script\getPadImg.py
```

2.2、将模型`base.torchscript`复制到`weights`文件夹下

2.3、使用`Icraft`编译模型

```bash
icraft compile configs\YoloV5_JDY_V2.0.ini
```

2.4、`Icraft.raw`转回`Pytorch.pth`

```bash
# 生成norm.pth
python script\raw2pth_norm.py

# 生成quantize.pth
python script\raw2pth_quantize.py
```

2.5、生成Icraft参数

```bash
# 生成ICRAFT.py
python script\create_ICRAFT.py
```



# 三、INQ

> 第三章所有命令均在`ICraft_yolov5`仓库中运行

3.1、切换到INQ分支

```bash
# 该分支模拟了Icraft将特征图归一化后的运算，与正常模型不通用
git checkout INQ
```

3.2、将ICRAFT.py复制到当前路径下；将norm.pth、quantize.pth复制到`runs/train/fuse/weights/`

3.3、测试norm精度

> 事实上由于添加了特征图量化算子，该步骤已经无法复现
>
> 除非你手动把models/common.py内FakeQuantize的forward改为return x
>
> 记得改回来

```bash
python3 val_norm.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.457
```

3.4、测试quantize精度

```bash
python3 val_quantize.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.244  ↓3.6%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.419  ↓3.8%
```

3.5、开始INQ

```bash
python3 train_INQ50.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ50

python3 train_INQ75.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ75

python3 train_INQ875.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ875

python3 train_INQ95.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ95

python3 train_INQ99.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ99
```

3.6、测试INQ精度

```bash
python3 val.py --data coco.yaml --weights runs/train/INQ50/weights/best.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.266  ↓1.4%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.444  ↓1.3%

python3 val.py --data coco.yaml --weights runs/train/INQ75/weights/best.pt  --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.266  ↓1.4%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.443  ↓1.4%

python3 val.py --data coco.yaml --weights runs/train/INQ875/weights/best.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.266  ↓1.4%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.444  ↓1.3%

python3 val.py --data coco.yaml --weights runs/train/INQ95/weights/best.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.266  ↓1.4%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.443  ↓1.4%

python3 val.py --data coco.yaml --weights runs/train/INQ99/weights/best.pt --iou-thres 0.65
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.263  ↓1.7%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.442  ↓1.5%
```

3.7、`Pytorch.pth`转回`Icraft.raw`

```bash
# 将在json&raw文件夹下生成YoloV5_quantized_INQ.raw
python script\pth2raw_quantize.py

# 复制一份json
cp json&raw\YoloV5_quantized.json json&raw\YoloV5_quantized_INQ.json

# 将json&raw\YoloV5_quantized_INQ.json内的"raw_md5"改为YoloV5_quantized_INQ.raw的md5值
```

3.8、效果测试

```bash
# Icraft浮点
python script\val_float.py
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.434

# Icraft定点-INQ前
python script\val_quant.py
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.226  ↓3.4%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.394  ↓4.0%

# Icraft定点-INQ后
python script\val_quant_INQ.py
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.233  ↓2.7%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.407  ↓2.7%
```

