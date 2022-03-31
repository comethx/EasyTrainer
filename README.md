# EasyTrainer
使用**几行代码**，快速训练并调用一个机器学习图像分类模型。

```python
from EasyTrainerCore import EasyTrain
if __name__ == "__main__":
    img = Image.open("your image path")
    model = EasyTrain.start()
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

在项目文件夹下，新建文件夹名为pictures，用于存放图像数据

<img src="https://rufuspic.oss-cn-chengdu.aliyuncs.com/markdown_imgs/image-20220331212737090.png" alt="image-20220331212737090" style="zoom: 10%;" />

将数据集放入pictures文件夹内，路径如图所示。注意，这里的label1，label2等标签可以为中文。比如，假设放入的是水果数据集，那么label1可以是“西瓜”，label2可以是“梨子”。

<img src="https://rufuspic.oss-cn-chengdu.aliyuncs.com/markdown_imgs/image-20220331213621738.png" alt="image-20220331213621738" style="zoom: 10%;" />

**至此，所有准备工作已经完成**

下面打开demo.py文件，即可运行示例。

## 模型保存并调用

训练开始时，EasyTrainer会自动在当前目录下生成一个weights文件夹，用于存放模型数据。weights文件夹存有下有模型检查点文件夹和模型原本的文件。

通过读取检查点的模型文件，可以快速实现模型调用并预测结果。下方代码块输出的result，和pictures下的文件夹名字保持一致。即，EasyModel帮我们自动进行了数字下标到真正标签名的替换工作。

```python
from EasyTrainerCore.Model import EasyModel
from PIL import Image

model = EasyModel("/weights/your_model_name/??_epoch.pth")
img = Image.open(img_path)
result = model.predict(img)
```

**请注意：上述所有操作，必须保证运行初始的路径在项目目录文件夹下，否则会引发路径错误。**

## TODO LIST

- [ ] 关于自定义训练过程参数的详细说明
- [ ] 英文文档
- [ ] 改进性能
