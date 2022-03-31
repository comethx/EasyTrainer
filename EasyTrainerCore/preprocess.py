import os
import glob

import random
from EasyTrainerCore.utils.to_json import init_label_to_name_json


def init_txt(init=False, split=0.8):
    traindata_path = 'pictures'
    labels = os.listdir(traindata_path)
    if not init:
        return len(labels)
    savepath = 'EasyTrainerCore/data/'
    print(
        "<EasyTrainer> Integrating pictures into txt in EasyTrainerCore/data/train.txt and EasyTrainerCore/data/val.txt")
    print("<EasyTrainer> " + str(len(labels)) + " categories detected")
    print("<EasyTrainer> Labels:" + str(labels))
    if os.path.exists(savepath + 'train.txt'):
        os.remove(savepath + 'train.txt')
    if os.path.exists(savepath + 'val.txt'):
        os.remove(savepath + 'val.txt')
    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(traindata_path, label, '*.jpg'))
        random.shuffle(imglist)
        trainlist = imglist[:int(split * len(imglist))]
        vallist = imglist[(int(split * len(imglist)) + 1):]
        with open(savepath + 'train.txt', 'a') as f:
            for img in trainlist:
                f.write(img + ',' + str(index))
                f.write('\n')

        with open(savepath + 'val.txt', 'a') as f:
            for img in vallist:
                f.write(img + ',' + str(index))
                f.write('\n')
    print("<EasyTrainer> Generating label_to_name json in EasyTrainerCore/data/label_to_name.json")
    init_label_to_name_json()
    return len(labels)
