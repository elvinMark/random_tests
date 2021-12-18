import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import timm

import sys
import numpy as np

from collections import OrderedDict

# set seed
torch.manual_seed(1234)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias != None:
            init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class resnet_basic(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(resnet_basic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                in_planes,
                                out_planes,
                                kernel_size=1,
                                stride=stride,
                                bias=False,
                            ),
                        ),
                        ("batch_norm", nn.BatchNorm2d(out_planes)),
                    ]
                )
            )

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu2(o + self.shortcut(x))
        return o


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet_CIFAR10(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_CIFAR10, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1
        )
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2
        )
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2
        )
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        # return nn.Sequential(*layers)
        return nn.Sequential(
            OrderedDict(
                [("wide_basic_%d" % i, layer) for (i, layer) in enumerate(layers)]
            )
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class Wide_ResNet_Tiny(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_Tiny, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1
        )
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2
        )
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2
        )
        self.layer4 = self._wide_layer(
            wide_basic, nStages[4], n, dropout_rate, stride=2
        )
        self.bn1 = nn.BatchNorm2d(nStages[4], momentum=0.9)
        self.linear = nn.Linear(nStages[4], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        # return nn.Sequential(*layers)
        return nn.Sequential(
            OrderedDict(
                [("wide_basic_%d" % i, layer) for (i, layer) in enumerate(layers)]
            )
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class Resnet_Imagenet(nn.Module):
    def __init__(self, block, num_blocks):
        super(Resnet_Imagenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_planes = 64

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 1000)

    def make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride))
        stride = 1
        for i in range(num_blocks - 1):
            layers.append(block(planes, planes, stride=stride))
        self.in_planes = planes
        return nn.Sequential(
            OrderedDict(
                [("resnet_basic_%d" % i, layer) for (i, layer) in enumerate(layers)]
            )
        )

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.max_pool1(o)

        o = self.layer1(o)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)

        o = self.avg(o)
        o = o.view((-1, self.in_planes))
        o = self.fc(o)

        return o


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        OrderedDict(
            [
                ("dense1", dense(dim, dim * expansion_factor)),
                ("gelu", nn.GELU()),
                ("dropout", nn.Dropout(dropout)),
                ("dense2", dense(dim * expansion_factor, dim)),
                ("dropout", nn.Dropout(dropout)),
            ]
        )
    )


def MLPMixer(
    *,
    image_size,
    channels,
    patch_size,
    dim,
    depth,
    num_classes,
    expansion_factor=4,
    dropout=0.0,
):
    assert (image_size % patch_size) == 0, "image must be divisible by patch size"
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        OrderedDict(
            [
                (
                    "rearrange",
                    Rearrange(
                        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                        p1=patch_size,
                        p2=patch_size,
                    ),
                ),
                ("linear", nn.Linear((patch_size ** 2) * channels, dim)),
                *[
                    (
                        f"depth_{i}",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "norm_linear_residual_1",
                                        PreNormResidual(
                                            dim,
                                            FeedForward(
                                                num_patches,
                                                expansion_factor,
                                                dropout,
                                                chan_first,
                                            ),
                                        ),
                                    ),
                                    (
                                        "norm_linear_residual_2",
                                        PreNormResidual(
                                            dim,
                                            FeedForward(
                                                dim,
                                                expansion_factor,
                                                dropout,
                                                chan_last,
                                            ),
                                        ),
                                    ),
                                ]
                            )
                        ),
                    )
                    for i in range(depth)
                ],
                ("layer_norm", nn.LayerNorm(dim)),
                ("mean_reduce", Reduce("b n c -> b c", "mean")),
                ("fc", nn.Linear(dim, num_classes)),
            ]
        )
    )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv2d",
                    nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
                ),
                ("gelu", nn.GELU()),
                ("bn", nn.BatchNorm2d(dim)),
                *[
                    (
                        f"depth_{i}",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "res",
                                        Residual(
                                            nn.Sequential(
                                                OrderedDict(
                                                    [
                                                        (
                                                            "conv2d",
                                                            nn.Conv2d(
                                                                dim,
                                                                dim,
                                                                kernel_size,
                                                                groups=dim,
                                                                padding="same",
                                                            ),
                                                        ),
                                                        ("gelu", nn.GELU()),
                                                        ("bn", nn.BatchNorm2d(dim)),
                                                    ]
                                                )
                                            )
                                        ),
                                    ),
                                    ("conv2d", nn.Conv2d(dim, dim, kernel_size=1)),
                                    ("gelu", nn.GELU()),
                                    ("bn", nn.BatchNorm2d(dim)),
                                ]
                            )
                        ),
                    )
                    for i in range(depth)
                ],
                ("adaptive2d", nn.AdaptiveAvgPool2d((1, 1))),
                ("flatten", nn.Flatten()),
                ("linear", nn.Linear(dim, n_classes)),
            ]
        )
    )


def create_model(args, pretrained=False):
    if args.arch == "WRN":
        if args.dataset == "CIFAR10":
            return Wide_ResNet_CIFAR10(28, 10, 0, 10)
        elif args.dataset == "TinyImagenet":
            return Wide_ResNet_Tiny(16, 5, 0, 200)
        else:
            return None
    elif args.arch == "Resnet18":
        if args.dataset == "Imagenet":
            return Resnet_Imagenet(resnet_basic, [2, 2, 2, 2])
        else:
            return None
    elif args.arch == "default_Resnet18":
        if args.dataset == "Imagenet":
            return models.resnet18(pretrained=pretrained)
        else:
            return None
    elif args.arch == "ConvMixer":
        if args.dataset == "CIFAR10":
            return ConvMixer(256, 8, n_classes=10, patch_size=2, kernel_size=5)
        else:
            return None
    elif args.arch == "MLPMixer":
        if args.dataset == "CIFAR10":
            return MLPMixer(
                image_size=32,
                channels=3,
                patch_size=8,
                dim=128,
                depth=10,
                num_classes=10,
            )
        else:
            return None
    elif args.arch == "ViT":
        if args.dataset == "Imagenet":
            return timm.models.vit_tiny_patch16_224(pretrained=pretrained)
    else:
        return None
