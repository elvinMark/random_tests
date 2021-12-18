import os
import sys
import random
import time
import warnings
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import sam

import wandb
import pandas as pd

from utils import *
from datasets import *
from models import *
from scheduler import *


parser = create_parser()


def main():
    args = parser.parse_args()
    args.distributed = True
    main_worker(args)


def main_worker(args):
    global best_acc

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        master_addr = os.getenv("MASTER_ADDR", default="localhost")
        master_port = os.getenv("MASTER_PORT", default="8888")
        method = "tcp://{}:{}".format(master_addr, master_port)
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
        ngpus_per_node = torch.cuda.device_count()
        device = rank % ngpus_per_node
        print(f"rank : {rank}    world_size : {world_size}")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            "nccl", init_method=method, world_size=world_size, rank=rank
        )
        args.rank = rank
        args.gpu = device
        args.world_size = world_size

    # load model
    print(f"=> load model '{args.arch}'")
    if args.model_path:
        model = create_model(args)

    else:
        print("Loading pretrained model")
        model = create_model(args, pretrained=True)
        model.eval()

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + args.world_size - 1) / args.world_size)
            process_group = torch.distributed.new_group(
                [i for i in range(args.world_size)]
            )
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.model_path:
        tmp_ = torch.load(args.model_path)
        model.load_state_dict(tmp_)
        model.eval()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    train_loader, test_loader, train_sampler, test_sampler = create_dataset(args)

    if args.rank == 0:
        print(args)

    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return

    if args.distributed:
        train_sampler.set_epoch(0)

    # get training accuracy and training loss
    if "train" in args.test_dataset:
        train_acc, train_loss = validate(train_loader, model, criterion, args)
    else:
        train_acc, train_loss = "--", "--"

    # evaluate on validation set
    if "test" in args.test_dataset:
        test_acc, test_loss = validate(test_loader, model, criterion, args)
    else:
        test_acc, test_loss = "--", "--"

    if args.rank == 0:
        print("---------DONE---------")
        print(f"train accuracy: {train_acc}, train loss:{train_loss}")
        print(f"test accuracy: {test_acc}, test loss:{test_loss}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    p = multiprocessing.Process()
    p.start()
    main()
