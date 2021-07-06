import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

mydatasets = ["CIFAR10"]


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

def create_mydataset(dataset_name,is_training=True):
    cifar10_path = os.getenv("CIFAR10_PATH")
    if dataset_name == "CIFAR10":
        if is_training:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    CIFAR10_MEAN,
                    CIFAR10_STD
                )
            ])
            train_dataset = datasets.CIFAR10(
                root=cifar10_path,
                train=True, 
                transform=train_transform,
                download=True
            )
            return train_dataset
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    CIFAR10_MEAN,
                    CIFAR10_STD
                )
            ])
        
            test_dataset = datasets.CIFAR10(
                root=cifar10_path,
                train=False,
                transform=test_transform,
                download=True)
            return test_dataset
    else:
        return None
