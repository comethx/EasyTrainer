import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from EasyTrainerCore.data.transform import get_transform


class SelfCustomDataset(Dataset):
    def __init__(self, label_file, imageset, picture_size):

        with open(label_file, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(','), f))
        self.img_aug = True
        if imageset == 'train':
            self.transform = get_transform(size=picture_size)
        else:
            self.transform = get_transform(size=picture_size)
        self.input_size = picture_size

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.img_aug:
            img = self.transform(img)


        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))

    def __len__(self):
        return len(self.imgs)


def get_train_dataloader_and_length(train_label_dir, picture_size, batch_size):
    train_datasets = SelfCustomDataset(train_label_dir, imageset='train', picture_size=picture_size)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, len(train_datasets)


def get_val_dataloader(val_label_dir, picture_size, batch_size):
    val_datasets = SelfCustomDataset(val_label_dir, imageset='test', picture_size=picture_size)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=2)
    return val_dataloader
