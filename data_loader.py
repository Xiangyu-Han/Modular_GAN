from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

# 预处理和加载数据
class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    """
     train_dataset的数据格式如下
     '000003.jpg', [True, False, False, False, True]],
    """

    # 数据的预处理
    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        # rstrip() 删除 string 字符串末尾的指定字符(默认为空格). 处理txt标签文件
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        # all_attr_names表示全部40种任务类别集合
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i  # eg:att2idx[3] = BAGS_Unser_Eyes
            self.idx2attr[i] = attr_name  # eg:idx2atrr[BAGS_Unser_Eyes] = 3

        lines = lines[2:]  # 取第3行开始的1和-1标签行
        random.seed(1234)
        # random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()  # ['085399.jpg', '-1', '1', '.....]
            filename = split[0]  # 文件名 '085399.jpg'
            values = split[1:]  # 图片对应的标签

            label = []  # [False, False, False, False, True]
            # self.selected_attrs表示我们训练选用的任务类别集合
            # 默认的是[‘Black_Hair’, ‘Blond_Hair’, ‘Brown_Hair’, ‘Male’, ‘Young’]
            for attr_name in self.selected_attrs:
                # 取出所需标签对应的idx,如Black_hair对应idx为8
                idx = self.attr2idx[attr_name]
                # 如果图片里有黑头发，把True添加到labels，没有添加False
                label.append(values[idx] == '1')

            if (i+1) > 89200 and i < 89217:  # 取2000张做测试集
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


# 创建一个data_loader
def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())  # 数据随机水平翻转
    transform.append(T.CenterCrop(crop_size))  # 从中间裁剪 size = (178,128)
    transform.append(T.Resize(image_size))  # 更改图片大小 128*128
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))  # 正则化
    transform = T.Compose(transform)  # 把transform整合到一起

    if dataset == 'CelebA':
        # dataset 是CelebA的一个对象
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    # DataLoader类中dataset参数必须是 data.Dataset 类
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader

if __name__ == '__main__':
    """
    test code
    """
    data_loader = get_loader(image_dir='D:/Work/Dataset/Celeba/img_aling_Celeba/img_align_celeba/', attr_path='D:/Work/Dataset/Celeba/img_aling_Celeba/list_attr_celeba.txt',
               selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair'])
    data_iter = iter(data_loader)
    x_fixed, c_org = next(data_iter)
    print(x_fixed)
    print(x_fixed.size())
    print(c_org)
    print(c_org.size())