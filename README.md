# 一、得到基准模型

1.1、克隆工程[ICraft_yolov5: Fine tuning YOLOv5 for Icraft](https://github.com/Qiuye12138/ICraft_yolov5)，当前处于`master`分支

> 第一章所有命令均在`ICraft_yolov5`仓库中运行

1.2、该仓库提供了一个预训练模型`yolov5n.pt`，测试其浮点精度（其他自行训练的模型也如此）

```bash
python3 val.py --data coco.yaml --weights yolov5n.pt --imgsz 640 --iou-thres 0.65

#   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
#   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.457
```

1.3、查看该模型推理效果

```bash
python3 detect.py --weights yolov5n.pt --source data/images/bus.jpg --imgsz 640 --conf-thres 0.5
```

1.3、导出模型，这会将模型中的`BatchNorm`层融合

```bash
# 将在yolov5n.pt附近将生成yolov5n.torchscript
python3 export.py --weights yolov5n.pt --include torchscript --imgsz 640
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

```bash
python3 train_fuse.py --data coco.yaml --cfg yolov5n.yaml --hyp data/hyps/stable_hyp.yaml --weights '' --batch-size 128 --imgsz 640 --device '1' --epochs 30 --name fuse

# --weights '' : 强制重新从yaml创建模型，而不是加载已有模型
```

1.6、测试新模型精度

```bash
python3 val.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --imgsz 640 --iou-thres 0.65

```



1.7、导出模型

```bash
# 将在best.pt附近将生成best.torchscript
python3 export.py --weights runs/train/fuse/weights/best.pt --include torchscript --imgsz 640

# 重命名为base.torchscript，作为后续实验的基准
mv runs/train/fuse/weights/best.torchscript runs/train/fuse/weights/base.torchscript
```

# 二、Icraft编译

> 第二章所有命令均在当前仓库中运行

2.1、生成测试图片

```bash
python .\script\getPadImg.py
```

2.2、将模型`base.torchscript`复制到`weights`文件夹下

2.3、使用`Icraft`编译模型

```bash
icraft compile .\configs\YoloV5_JDY_V2.0.ini
```

2.4、查看模型推理效果

```bash
python .\script\detect_float.py	# Icraft-浮点
python .\script\detect_quant.py	# Icraft-定点
```

2.5、测试`Icraft`浮点精度

```bash
python .\script\val_float.py

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.434
```

2.6、测试`Icraft`定点精度

```bash
python .\script\val_quant.py

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.226  ↓3.4%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.394  ↓4.0%
```

2.7、`Icraft.raw`转回`Pytorch.pth`

```bash
# 生成norm.pth
python .\script\raw2pth_norm.py

# 生成quantize.pth
python .\script\raw2pth_quantize.py
```

2.8、生成Icraft参数

```bash
# 生成ICRAFT.py
python .\script\create_ICRAFT.py
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
python3 val_norm.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --imgsz 640 --iou-thres 0.65

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.280
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.457
```

3.4、测试quantize精度

```bash
python3 val_quantize.py --data coco.yaml --weights runs/train/fuse/weights/best.pt --imgsz 640 --iou-thres 0.65

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.244  ↓3.6%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.419  ↓3.8%
```

3.5、INQ50

```bash
# INQ50
python3 train_INQ50.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --imgsz 640 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ50

python3 val.py --data coco.yaml --weights runs/train/INQ50/weights/best.pt --imgsz 640 --iou-thres 0.65

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.266  ↓1.4%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.444  ↓1.3%
```

3.6、INQ75

```bash
# INQ75
python3 train_INQ75.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --imgsz 640 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ75

python3 val.py --data coco.yaml --weights runs/train/INQ75/weights/best.pt --imgsz 640 --iou-thres 0.65

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265  ↓1.5%
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.442  ↓1.5%
```

3.7、INQ875

```bash
python3 train_INQ875.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128 --imgsz 640 --device '1' --epochs 15 --hyp data/hyps/stable_hyp.yaml --name INQ875
```



3.4、`Pytorch.pth`转回`Icraft.raw`

