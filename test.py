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
import model_loader
import math
import sam

import wandb
import pandas as pd

from utils import *
from datasets import *
from models import *
from scheduler import *


parser = create_parser()

best_acc = 0

data_buffer = []


def main():
    args = parser.parse_args()

    if args.warmup > 0:
        for i in range(len(args.milestones)):
            args.milestones[i] += args.warmup

    args.epochs += args.warmup

    if args.T_max == -1:
        args.T_max = args.epochs

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("slow work because of to apply seed setting")

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

    config = {}
    if args.rank == 0:
        # Init wandb
        wandb.init(
            project=args.project,
            config={
                "global_batch_size": args.batch_size,
                "local_batch_size": int(args.batch_size / args.world_size),
                "np": args.world_size,
                "weight_decay": args.weight_decay,
                "initial_learn_rate": args.lr,
                "momentum": args.momentum,
                "nbs": args.nbs,
            },
            name=args.experiment,
        )
        config = wandb.config

    # create model
    # arch = "WRN-28-10"
    print("=> creating model '{}'".format(args.arch))
    # model = model_loader.WRN(28, 10, 0, 10)
    model = create_model(args)
    # init model layers
    # model.apply(model_loader.conv_init)
    model.apply(conv_init)

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

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = sam.SAM(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nbs=args.nbs,
    )

    cudnn.benchmark = True

    # Data loading code
    # train_loader, test_loader, train_sampler, test_sampler = load_cifar10(args)
    train_loader, test_loader, train_sampler, test_sampler = create_dataset(args)

    if args.rank == 0:
        print(args)

    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return

    # learning_rate = create_lr_scheduler(args)
    learning_rate = create_warmup(create_lr_scheduler(args), args)

    if args.rank == 0:
        # make models save dir if not exist
        os.makedirs(f"./trained_models/WRN-28-10_with_original_SAM", exist_ok=True)

        # prepare for saving the model
        save_list = []
        save_list.append(f"epochs={args.epochs}")
        save_list.append(f"lr={args.lr}")
        save_list.append(f"momentum={args.momentum}")
        save_list.append(f"weight_decay={args.weight_decay}")
        save_list.append(f"nbs={args.nbs}")
        save_list.append(f'global_batch={config["global_batch_size"]}')
        save_list.append(f"local_batch={args.batch_size}")
        model_path = "".join(
            [
                "./trained_models/WRN-28-10_with_original_SAM/model_",
                "_".join(save_list),
                ".ckpt",
            ]
        )
        optim_path = "".join(
            [
                "./trained_models/WRN-28-10_with_original_SAM/optim_",
                "_".join(save_list),
                ".ckpt",
            ]
        )

    if args.rank == 0:
        wandb.log({"model_path": model_path, "optim_path": optim_path})

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate(args.lr, epoch + 1)

        # train for one epoch
        train_acc, train_loss = train(
            train_loader, model, criterion, optimizer, epoch, args
        )

        # evaluate on validation set
        test_acc, test_loss = validate(test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if is_best and args.rank == 0:
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optim_path)

        if args.rank == 0:
            data_buffer.append(
                [
                    learning_rate(args.lr, epoch + 1),
                    train_acc.item(),
                    train_loss,
                    test_acc.item(),
                    test_loss,
                ]
            )
            wandb.log(
                {
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "epoch_lr": learning_rate(args.lr, epoch + 1),
                }
            )

    if args.rank == 0:
        print("---------DONE---------")
        print("FINAL BEST test_accuracy: {} %".format(best_acc))
        wandb.log({"best_test_acc": best_acc})

        df = pd.DataFrame(
            data=data_buffer,
            columns=["lr", "train_acc", "train_loss", "test_acc", "test_loss"],
        )

        df.to_csv("results_csv/" + args.experiment + ".csv")

        # save parameters, model_path
        log_dict = {"model_path": model_path, "optim_path": optim_path}
        print(log_dict)
        print(args)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    p = multiprocessing.Process()
    p.start()
    main()
