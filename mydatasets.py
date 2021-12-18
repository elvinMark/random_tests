import torchvision
import torch

ROOT_PATH = "../data/"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Using the same mean and standard deviation as in CIFAR10
CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_STD = (0.2023, 0.1994, 0.2010)

datasets_dict = {
    "MNIST" : {
        "loader" : torchvision.datasets.MNIST,
        "transform" : torchvision.transforms.ToTensor(),
        "test_transform" : torchvision.transforms.ToTensor()
    },
    "CIFAR10": {
        "loader" : torchvision.datasets.CIFAR10,
        "transform" : torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32,padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
        ]),

        "test_transform" : torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN,CIFAR10_STD)
        ])
    },
    "CIFAR100": {
        "loader" : torchvision.datasets.CIFAR100,
        "transform" : torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32,padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR100_MEAN,CIFAR100_STD)
        ]),

        "test_transform" : torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR100_MEAN,CIFAR100_STD)
        ])
    }
}

def create_dataloaders(args):
    train_ds = datasets_dict[args.dataset]["loader"](ROOT_PATH,train=True,download=True,transform=datasets_dict[args.dataset]["transform"])
    test_ds = datasets_dict[args.dataset]["loader"](ROOT_PATH,train=False,download=True,transform=datasets_dict[args.dataset]["test_transform"])

    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds,batch_size=args.batch_size)
        
    return train_dl, test_dl
        
