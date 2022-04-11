# EasyTrainer

[🔎Chinese documentation](https://github.com/comethx/EasyTrainer/blob/main/README_CN.md)

Quickly train and invoke a machine learning image classification model with **a few lines of code**.

```python
from EasyTrainerCore import EasyTrain

if __name__ == "__main__":
    model = EasyTrain.start()
    img = Image.open("your image path")
    result = model.predict(img)
```

This project is a further encapsulation of Pytorch and is aimed at helping beginners of deep learning. My purpose is to reduce the time for debugging code, so that everyone can focus on model tuning and participate in optimization strategy selection.

## Getting started

### Environment preparation

First clone this repository

```shell
git clone https://github.com/comethx/EasyTrainer.git
```

Then install the relevant environment(If you already have pytorch and its components installed, you can skip this step)

```shell
pip install -r requirements.txt
```

### dataset preparation

Go to the project folder, create a new folder named **pictures** to store image data

```shell
cd EasyTrainer && mkdir pictures
```

Put the dataset into the pictures folder, and the path is as shown in the figure. For example, if you put a fruit dataset, then label1 can be "watermelon", and label2 can be "pear".

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

**So far, all preparations have been completed**

Open the demo.py file below and run the example code to train the model.

```python
from EasyTrainerCore import EasyTrain

if __name__ == "__main__":
    EasyTrain.start()
```

## Training with parameter customization

If you have successfully run the demo.py file, you can try further customizations. Open the demo_plus file and modify various parameters of EasyTrain.start to implement custom model training.

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

## Use the trained model

After training starts, when the model needs to be saved, EasyTrainer will automatically generate a **weights** folder in the current directory to store the model data. The weights folder contains the model checkpoint folder and the original files of the model.

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

By loading the checkpointed model file, you can quickly use model to predict results. **The result output by the code block below is the same as the folder name under pictures**. That is, EasyModel helps us to automatically replace the number subscript to the real tag name.

```python
from EasyTrainerCore.Model import EasyModel
from PIL import Image

model = EasyModel("/weights/your_model_name/??_epoch.pth")
img = Image.open(img_path)
result, confidence = model.predict(img)
```

**Please note: For all the above operations, you must ensure that the initial path to run is under the project directory folder, otherwise a path error will occur. **

## TODO LIST

- [ ] Detailed description
- [ ] Improve code comments
- [x] English Documentation
- [ ] Improve performance

