import random
import tarfile
from typing import Optional, Callable
from PIL import Image
from torch.utils.data import Dataset, get_worker_info, DataLoader
from torchvision.transforms import transforms

from collections import defaultdict


class SPDATA_trip(Dataset):
    seed = 27

    def __init__(self, tar_path, sep, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 tv_rate=0.8, train: bool = True):
        # dataTarFile是一个.tgz文件的路径，如：C:\Users\shuke\Downloads\oxbuild_images.tgz
        # 将文件名按照sep来分割，区sep[0]为标签，如：oxc1_000001.jpg，sep为'_'，则标签为oxc1
        # tv_rate是训练集和验证集的比例，如：0.8，表示训练集占80%，验证集占20%
        super().__init__()

        self.tar_obj = {}
        self.sep = sep
        self.tar_path = tar_path
        self.transform = transform
        self.target_transform = target_transform
        self.tv_rate = tv_rate
        self.train = train
        # 打开Tar文件
        with tarfile.open(self.tar_path) as tar:
            # 获取Tar文件中的所有文件名
            self.names = [x.name for x in tar.getmembers()]
        # 生成标签列表
        labels = set([x.split(self.sep)[0] for x in self.names])
        labels = list(labels)
        labels.sort()
        self.labels = {label: idx for idx, label in enumerate(labels)}
        # 打乱数据集
        random.Random(self.seed).shuffle(self.names)
        # 划分训练集和验证集的文件名
        if train:
            self.names = self.names[:int(len(self.names) * tv_rate)]
        else:
            self.names = self.names[int(len(self.names) * tv_rate):]

        self.data = defaultdict(list)

        for i in self.names:
            self.data[self.labels[i.split(self.sep)[0]]].append(i)

    def __len__(self):
        return len(self.names) // 3

    def label2index(self):
        return self.labels

    def index2label(self):
        lb = {}
        for k, v in self.labels.items():
            lb[v] = k
        return lb

    def __getitem__(self, idx):
        pos_l, neg_l = random.sample(list(self.data.keys()), 2)

        anc, pos = random.sample(self.data[pos_l], 2)
        neg = random.choice(self.data[neg_l])

        pos = Image.open(self.read_file(pos))
        anc = Image.open(self.read_file(anc))
        neg = Image.open(self.read_file(neg))

        if self.transform is not None:
            pos = self.transform(pos)
            anc = self.transform(anc)
            neg = self.transform(neg)

        return (anc, pos, neg), (pos_l, pos_l, neg_l)

    def read_file(self, file_name):
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_obj:
            # self.tar_obj[worker][0]为tarfile
            # self.tar_obj[worker][1]为{文件名: 地址}
            self.tar_obj[worker] = [tarfile.open(self.tar_path)]
            members = self.tar_obj[worker][0].getmembers()
            self.tar_obj[worker].append({m.name: m for m in members})
        return self.tar_obj[worker][0].extractfile(self.tar_obj[worker][1][file_name])


if __name__ == '__main__':
    import multiprocessing

    train_set = SPDATA(r"../data/inshop.tar", sep='-', train=True,
                       transform=transforms.Compose(
                           [transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=4, num_workers=2,
                              shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True)
    for i in train_loader:
        print(len(i[0]))
        exit()
