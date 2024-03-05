import random
import tarfile
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset
import os

from torchvision.transforms import transforms


class SPDATA(Dataset):
    seed = 27

    def __init__(self, labelPath, imgPath, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__()

        self.labelPath = labelPath
        self.imgPath = imgPath
        self.transform = transform
        self.target_transform = target_transform
        self.img = []

        labels = set()
        with open(self.labelPath) as f:
            lines = f.readlines()
        for i in lines:
            path, label = i.split(' ')
            labels.add(label[:-1])
            self.img.append((path, label[:-1]))
        labels = list(labels)
        labels.sort()
        self.labels = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        path, label = self.img[idx]
        label = self.labels[label]
        img = Image.open(os.path.join(self.imgPath, path))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


if __name__ == '__main__':
    import torch
    from torchvision import datasets, transforms

    # 加载数据集
    transform = transforms.Compose([transforms.ToTensor()])
    # trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = SPDATA(r'C:\Users\29776\proj\py\image_retrieval\train.txt',
                      r'C:\Users\29776\proj\py\image_retrieval\train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    # Mean:  tensor([0.7503, 0.7172, 0.7050])
    # Std:  tensor([0.2935, 0.3060, 0.3119])

    # Mean:  tensor([0.7512, 0.7186, 0.7061])
    # Std:  tensor([0.2925, 0.3047, 0.3102])
    # 计算均值和标准差
    data = next(iter(trainloader))[0]
    mean = data.mean(dim=[0, 2, 3])
    std = data.std(dim=[0, 2, 3])

    print('Mean: ', mean)
    print('Std: ', std)
