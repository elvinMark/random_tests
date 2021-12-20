import torch
from torchvision import transforms, datasets
import PIL
from PIL import Image

mnist_root = "./mnist_data"

cifar10_traindir = "./cifar10_data/train"
cifar10_valdir = "./cifar10_data/validation"

tiny_traindir = "../data/tiny-imagenet-200/train"
tiny_valdir = "../data/tiny-imagenet-200/val"

imagenet_traindir = "/mnt/nfs/datasets/ILSVRC2012/train"
imagenet_valdir = "/mnt/nfs/datasets/ILSVRC2012/val"


def load_mnist(args):
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        mnist_root, train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.MNIST(
        mnist_root, train=False, download=True, transform=train_transform
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=args.world_size, rank=args.rank
        )
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, test_loader, train_sampler, test_sampler


def load_cifar10(args):
    # データ正規化
    train_form = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]  # RGB 平均  # RGB 標準偏差
            ),
        ]
    )

    test_form = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]  # RGB 平均  # RGB 標準偏差
            ),
        ]
    )

    # load CIFAR10 data
    train_dataset = datasets.CIFAR10(  # CIFAR10 default dataset
        root=cifar10_traindir,  # rootで指定したフォルダーを作成して生データを展開。これは必須。
        train=True,  # 学習かテストかの選択。これは学習用のデータセット
        transform=train_form,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=cifar10_valdir,
        train=False,  # 学習かテストかの選択。これはテスト用のデータセット
        transform=test_form,
        download=True,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=args.world_size, rank=args.rank
        )
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, test_loader, train_sampler, test_sampler


def load_tinyimagenet(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(root=tiny_traindir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=tiny_valdir, transform=test_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=args.world_size, rank=args.rank
        )
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, test_loader, train_sampler, test_sampler


def load_imagenet(args):
    # train_transform = transforms.Compose([
    #     transforms.Resize(size=256,interpolation=PIL.Image.BILINEAR),
    #     transforms.CenterCrop(size=(224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    # ])

    mean_ = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]

    # mean_ = [0, 0, 0]
    # std_ = [1, 1, 1]

    train_transform = transforms.Compose(
        [
            transforms.Resize(size=256, interpolation=PIL.Image.BILINEAR),
            transforms.RandomCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(size=256, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_, std=std_),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=imagenet_traindir, transform=train_transform
    )
    test_dataset = datasets.ImageFolder(root=imagenet_valdir, transform=test_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=args.world_size, rank=args.rank
        )
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, test_loader, train_sampler, test_sampler


def create_dataset(args):
    if args.dataset == "MNIST":
        return load_mnist(args)
    elif args.dataset == "CIFAR10":
        return load_cifar10(args)
    elif args.dataset == "TinyImagenet":
        return load_tinyimagenet(args)
    elif args.dataset == "Imagenet":
        return load_imagenet(args)
    else:
        return None
