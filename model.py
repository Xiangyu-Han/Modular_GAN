import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Encoder(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Encoder, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append((nn.ReLU(inplace=True)))

        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        # Residual Blocks * 6

        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)

        return h


class Transformer(nn.Module):
    def __init__(self, conv_dim=256, c_dim=5, repeat_num=6):
        super(Transformer, self).__init__()

        layers = []
        layers.append(nn.Conv2d(conv_dim+c_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))

        self.main = nn.Sequential(*layers)

        # 注意力层生成mask
        self.attention = nn.Sequential(
            nn.Conv2d(conv_dim, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x, c):
        # 把c的维度复制为和x一样的维度，以供拼接
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        # 拼接
        h = torch.cat((x, c), dim=1)

        f = self.main(h)
        # 联合mask和经过残差层的输出
        g = (1 + self.attention(f)) / 2
        return g * f + (1 - g) * x


class Reconstructor(nn.Module):
    """
    Reconstructor Module networks
    """
    def __init__(self, conv_dim=256):
        super(Reconstructor, self).__init__()

        layers = []

        # up-sampling layers
        for _ in range(2):
            layers.append(nn.ConvTranspose2d(conv_dim, conv_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(conv_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            conv_dim //= 2

        # convlutional layer
        layers.append(nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        # layers.append((nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)))
        layers.append(nn.LeakyReLU(0.01))

        cur_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(cur_dim, cur_dim*2, kernel_size=4, stride=2, padding=1))
            # layers.append(nn.InstanceNorm2d(cur_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.LeakyReLU(0.01))
            cur_dim = cur_dim * 2

        self.main = nn.Sequential(*layers)

        kernel_size = int(image_size / np.power(2, repeat_num))

        self.conv1 = nn.Conv2d(cur_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cur_dim, c_dim, kernel_size=image_size//2**repeat_num, bias=False)

    def forward(self, x):

        h = self.main(x)

        out_src = self.conv1(h)
        out_cls = self.conv2(h)

        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

'''
# 输入的x尺寸是[16,3,128,128],c尺寸是[16,5]
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        # 第一个卷积层,输入为图像和label的串联,3表示图像为3通道,c_dim为label的维度,
        layers = []
        # [Conv2d(8, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)]
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # InstanceNorm层
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        # 2个卷积层,stride=2,即下采样
        curr_dim = conv_dim  # 这时候的64个维度
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
            # 经过两次循环，这时 curr_dim 的维度为256

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        # 最后的维度为3维
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):  # 定义计算的过程
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        # [16,5,1,1]
        c = c.view(c.size(0), c.size(1), 1, 1)
        # 沿着指定的维度重复tensor [16,5,128,128]
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)  # 输入图像x,label向量c,串联
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim  # 2048
        for i in range(1, repeat_num):  # 循环5次
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))  # 128/2^6 = 2
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        # torch.Size([16, 3, 128, 128])
        h = self.main(x)
        # [16, 1, 2, 2]
        out_src = self.conv1(h)
        # [16, 5, 1, 1]
        out_cls = self.conv2(h)
        # [16, 5]
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))   
'''

if __name__ == '__main__':
    """
    test code
    """
    # model structure
    print("Encoder")
    E = Encoder().cuda()
    print(E)
    print('\n')

    print("Transformer")
    T = Transformer().cuda()
    print(T)
    print('\n')

    print("Reconstructor")
    R = Reconstructor().cuda()
    print(R)
    print('\n')

    print("Discriminator")
    D = Discriminator().cuda()
    print(D)
    print('\n')

    # tensor flow
    x = torch.randn(8, 3, 128, 128).cuda()
    c = torch.randn(8, 5).cuda()
    print("The size of input image: {}".format(x.size()))
    print("The size of input label: {}".format(c.size()))

    out = E(x)
    print("The size of Encoder output: {}".format(out.size()))

    out = T(out, c)
    print("The size of Transformer output: {}".format(out.size()))

    out = R(out)
    print("The size of Reconstructor output: {}".format(out.size()))

    out_src, out_cls = D(x)
    print("The size of src out: {}".format(out_src.size()))
    print("The size of cls out: {}".format(out_cls.size()))