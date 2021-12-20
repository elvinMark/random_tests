from timm.data.parsers import parser
import torch
from torchvision import transforms, datasets
import PIL
from PIL import Image
from timm.data.parsers.parser_image_in_tar import ParserImageInTar
from timm.data.parsers import create_parser
from timm.data import ImageDataset
import logging
import pickle

mnist_root = "./mnist_data"

cifar10_traindir = "./cifar10_data/train"
cifar10_valdir = "./cifar10_data/validation"

tiny_traindir = "../data/tiny-imagenet-200/train"
tiny_valdir = "../data/tiny-imagenet-200/val"

imagenet_traindir = "/mnt/nfs/datasets/ILSVRC2012/train"
imagenet_valdir = "/mnt/nfs/datasets/ILSVRC2012/val"

imagenet_tar_traindir = "/mnt/nfs/datasets/ILSVRC2012/ILSVRC2012_img_train.tar"
imagenet_tar_valdir = "/mnt/nfs/datasets/ILSVRC2012/ILSVRC2012_img_val.tar"

IMAGENET_VAL_ARR = "../data/val_arr.pkl"


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
                [0.4914, 0.4822, 0.4465], [
                    0.2023, 0.1994, 0.2010]  # RGB 平均  # RGB 標準偏差
            ),
        ]
    )

    test_form = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [
                    0.2023, 0.1994, 0.2010]  # RGB 平均  # RGB 標準偏差
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=tiny_traindir, transform=train_transform)
    test_dataset = datasets.ImageFolder(
        root=tiny_valdir, transform=test_transform)

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
    test_dataset = datasets.ImageFolder(
        root=imagenet_valdir, transform=test_transform)

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


_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50


class ImageNetTestDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(
                parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(
                f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        else:
            tmp_ = self.parser.filename(index)
            fn, _ = tmp_.split()
            target = self.target_transform[int(fn[16:23]) - 1]
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


def load_imagenet_tar(args):
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

    train_dataset = ImageDataset(
        root=imagenet_traindir, parser=ParserImageInTar(imagenet_traindir), transform=train_transform
    )

    with open(IMAGENET_VAL_ARR, "rb") as f:
        val_arr = pickle.load(f)

    test_dataset = ImageNetTestDataset(
        root=imagenet_valdir, parser=ParserImageInTar(imagenet_valdir), transform=test_transform, target_transform=val_arr)

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
    elif args.dataset == "ImagenetTar":
        return load_imagenet_tar(args)
    else:
        return None
