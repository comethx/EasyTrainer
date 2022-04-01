# EasyTrainer
使用**几行代码**，快速训练并调用一个机器学习图像分类模型。

```python
from EasyTrainerCore import EasyTrain

if __name__ == "__main__":
    model = EasyTrain.start()
    img = Image.open("your image path")
    result = model.predict(img)
```

本项目是对Pytorch的进一步封装，面向**深度学习的初学者**，目的是减少大家调通代码的时间，让大家专注于模型调参与优化策略选择。

## 快速开始

### 环境搭建

首先克隆本仓库

```
git clone https://github.com/comethx/EasyTrainer.git
```

然后安装相关环境

```
pip install -r requirements.txt
```

### 数据准备

进入项目文件夹下，**新建文件夹名为pictures**，用于存放图像数据

```shell
cd EasyTrainer && mkdir pictures
```

将数据集放入pictures文件夹内，路径如图所示。注意，这里的label1，label2等标签可以为中文。比如，假设放入的是水果数据集，那么label1可以是“西瓜”，label2可以是“梨子”。

```
│  demo.py
│  demo_plus.py
│  LICENSE
│  README.md
│  requirements.txt
│
├─EasyTrainerCore
└─pictures
    ├─label1
    │      fimg_1491.jpg
    │      fimg_1492.jpg
    │      fimg_1494.jpg
    │      fimg_1495.jpg
    │	   . . . .
    ├─label2
    │      baidu000000.jpg
    │      baidu000001.jpg
    │      baidu000003.jpg
    │      baidu000004.jpg
    │	   . . . .
    └─label3
            baidu000000.jpg
            baidu000001.jpg
            baidu000002.jpg
            baidu000003.jpg
 			. . . .
```

**至此，所有准备工作已经完成**

下面打开demo.py文件，运行实例代码，即可训练模型。

```python
from EasyTrainerCore import EasyTrain

if __name__ == "__main__":
    EasyTrain.start()
```

## 训练参数自定义

若已经成功运行了demo.py文件，则可以尝试进一步的自定义操作。打开demo_plus文件，修改EasyTrain.start的各种参数，即可实现自定义模型训练。

```python
from EasyTrainerCore import EasyTrain

if __name__ == "__main__":
    # after training, the EasyTrain.start() will return the latest model
    model = EasyTrain.start(

        gpu_nums=1,  # 0: using cpu to train, 1: using gpu to train, more than 1: using multi-gpu to train (default: 0)

        model_name="mobilenetv2",  # choose the model, you can choose from the list (default: efficientnet-b3)

        # 'resnext101_32x8d'
        # 'resnext101_32x16d',
        # 'resnext101_32x48d',
        # 'resnext101_32x32d',
        # 'resnet50',
        # 'resnet101',
        # 'densenet121',
        # 'densenet169',
        # 'mobilenetv2',
        # 'efficientnet-b0' ~ 'efficientnet-b8'

        froze_front_layers=True,  # To freeze the parameters of front layers (default: False)

        lr=1e-1,  # learning rate (default: 1e-2)
        lr_adjust_strategy="cosine",  # "cosine" or "step" (default: None)
        optimizer="Adam",  # SGD or Adam (default: Adam)
        loss_function="CrossEntropyLoss",
        # ↑ CrossEntropyLoss or FocalLoss or SoftmaxCrossEntropyLoss (default: CrossEntropyLoss)
        train_and_val_split=0.8,  # train and val split ratio (default: 0.8)
        picture_size=256,  # the picture size of the model (default: 64)
        batch_size=64,  # batch size for training (default: 64)

        resume_epoch=0,  # resume training from last_epoch (default: 0)
        max_epoch=3,  # max epoch for training (default: 10)
        save_sequence=2  # save model every n epochs (default: 2)
    )

```

## 模型保存并调用

训练开始后，当需要保存模型时，EasyTrainer会自动在当前目录下生成一个weights文件夹，用于存放模型数据。weights文件夹存有下有模型检查点文件夹和模型原本的文件。

```
├─EasyTrainerCore
├─pictures
└─weights
    │  efficientnet-b3-5fb5a3c3.pth
    │  mobilenet_v2-b0353104.pth
    │
    ├─efficientnet-b3
    │      epcoh_5.pth
    │
    └─mobilenetv2
            epcoh_10.pth
            epcoh_5.pth
```

通过读取检查点的模型文件，可以快速实现模型调用并预测结果。**下方代码块输出的result和pictures下的文件夹名字保持一致**。即EasyModel帮我们自动进行了数字下标到真正标签名的替换工作。

```python
from EasyTrainerCore.Model import EasyModel
from PIL import Image

model = EasyModel("/weights/your_model_name/??_epoch.pth")
img = Image.open(img_path)
result, confidence = model.predict(img)
```

**请注意：上述所有操作，必须保证运行初始的路径在项目目录文件夹下，否则会引发路径错误。**

## TODO LIST

- [ ] 详细说明
- [ ] 英文文档
- [ ] 改进性能
