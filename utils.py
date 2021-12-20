import argparse
import torch
import torch.distributed as dist
import time
import math
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class SmallIterator:
    def __init__(self, x, y, bs):
        self.x = x
        self.y = y
        self.curr = 0
        self.bs = bs  # local batch_size
        self.tot_size = len(self.x)  # global batch_size
        self.size = math.ceil(self.tot_size / self.bs)  # 4

    def __iter__(self):
        self.curr = 0
        return self

    def __len__(self):
        return self.size

    def __next__(self):
        if self.curr < self.tot_size:
            tmp = self.curr
            self.curr += self.bs
            return self.x[tmp : self.curr], self.y[tmp : self.curr]

        else:
            self.curr = 0
            raise StopIteration


def create_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch test distributed SAM Training"
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 5e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--nbs",
        "--neighborhood-size",
        default=0.05,
        type=float,
        metavar="S",
        help="neighborhood size",
        dest="nbs",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 30)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=1234, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["MNIST", "CIFAR10", "TinyImagenet", "Imagenet", "ImagenetTar"],
        help="Dataset to be used",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="WRN",
        choices=[
            "WRN",
            "Resnet18",
            "default_Resnet18",
            "Conv",
            "MLPMixer",
            "ConvMixer",
            "ViT",
        ],
        help="Architecture to be used",
    )
    parser.add_argument(
        "--project", type=str, default="SAM_tests_32768", help="Name of the project"
    )
    parser.add_argument(
        "--experiment", type=str, default="experiment", help="Name of the experiment"
    )

    parser.add_argument(
        "--sched",
        type=str,
        default="multistep",
        help="specify the type of scheduler to be used",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="specify the gamma value used in various schedulers. i.e. step, multistep",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=50,
        help="specify the step sized used in the step scheduler",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        default=[60, 120, 160],
        nargs="+",
        help="specify the type of scheduler to be used",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=0.0,
        help="specify the eta min used in the cosine scheduler",
    )
    parser.add_argument(
        "--T_max",
        type=int,
        default=-1,
        help="specify the T max used in the cosine scheduler",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="specify for how many epochs to warm up"
    )
    parser.add_argument(
        "--lower_lr",
        type=float,
        default=0.05,
        help="specify the lower lr to be used in the warm up",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="specify the path to the model to be loaded (use just for test_model)",
    )

    parser.add_argument(
        "--optim", type=str, default="sam", help="specify the optimizer to be used"
    )

    parser.add_argument(
        "--range",
        type=float,
        nargs="+",
        default=(-0.5, 0.5),
        help="specify the range of the x and y",
    )

    parser.add_argument(
        "--pca",
        type=str,
        default=None,
        help="specify the directory where the pca directions are stored",
    )

    parser.add_argument(
        "--test-dataset",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="specify whether to use train dataset and test dataset or just one of them for the model testing",
    )

    return parser


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_model_buffers(model):
    ret = {}
    for k, v in model.state_dict().items():
        if "mean" in k or "var" in k or "num_batches_tracked" in k:
            ret["model." + k + ".data"] = v.data.clone()
    return ret


def load_model_buffers(model, buf_dict):
    for k, v in buf_dict.items():
        exec("%s = v" % (k))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # save w_t params
        w_t = [
            param.clone().detach().requires_grad_(True) for param in model.parameters()
        ]

        # reset gradient
        optimizer.zero_grad()

        # compute output and normal_grad each sam_local_batch without all-reduce normal_grad
        # normal_gradはsam_bnごとに計算し，それぞれw_advを出すのに使うので，この時の同期は切断(all-reduceしない)
        # この時のbatch normの平均と分散は．globalにsync (Sync Batch Normを使用している)
        # ここでのbatch normの平均と分散を更新に反映
        # ここのforward計算前にはw_tは揃っているはず．(normal gradを用いたstep更新では各プロセスは全く同じ計算をしているはず．)

        small_ds = SmallIterator(images, target, 256)
        loss = torch.tensor(0.0)
        output = torch.tensor([])
        num_small_batch = len(small_ds)
        ret = None
        if args.gpu is not None:
            loss = loss.cuda(args.gpu)
            output = output.cuda(args.gpu)
        with model.no_sync():
            for (small_images, small_target) in small_ds:
                small_output = model(small_images)
                output = torch.cat((output, small_output))
                # calc normal_grad
                small_loss = criterion(small_output, small_target)
                small_loss /= num_small_batch
                small_loss.backward()
                if ret is None:
                    ret = save_model_buffers(model)
                    for k in ret:
                        if not "num_batches" in k:
                            ret[k] /= num_small_batch
                else:
                    tmp_ret = save_model_buffers(model)
                    for k in tmp_ret:
                        if not "num_batches" in k:
                            ret[k] += tmp_ret[k] / num_small_batch
                loss += small_loss

            load_model_buffers(model, ret)
            # compute w_adv from normal_grad
            optimizer.step_calc_w_adv()

        acc1 = accuracy(output, target)[0]
        # save model buffers (for Sync Batch Norm info)
        buf_list = save_model_buffers(model)

        # reset gradient
        optimizer.zero_grad()

        last_small_idx = num_small_batch - 1
        ret = None
        for small_idx, (small_images, small_target) in enumerate(small_ds):
            # この時のbatch normの平均と分散は．globalにsync (Sync Batch Normを使用している)
            # ここでのbatch normの平均と分散の更新は反映しない．(後に430行目でresetする．)
            if small_idx != last_small_idx:
                with model.no_sync():
                    small_output = model(small_images)

                    # calc sam loss (each sam batch)
                    loss_sam = criterion(small_output, small_target)

                    loss_sam /= num_small_batch

                    # calc global sam_grad (with all-reduce sam_grad each sam batch)
                    loss_sam.backward()
            else:
                small_output = model(small_images)

                # calc sam loss (each sam batch)
                loss_sam = criterion(small_output, small_target)

                loss_sam /= num_small_batch

                # calc global sam_grad (with all-reduce sam_grad each sam batch)
                loss_sam.backward()

        # load w_t weights params (without gradient information)
        # update weights params from sam grad
        # 元の位置(w_t)からglobal sam_gradを用いて更新！
        optimizer.load_original_params_and_step(w_t)

        # この処理でacc1は平均されて処理されるはず！
        # この処理でlossも平均されて処理されるはず！
        if args.distributed:
            size = float(args.world_size)
            dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
            acc1 /= size
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= size

        # update loss and acc info
        top1.update(acc1[0], images.size(0))
        losses.update(loss.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0:
            if i % args.print_freq == 0:
                progress.display(i)

        # load model buffers (reset BN info because computed BN info again)
        load_model_buffers(model, buf_list)

    return top1.avg, losses.avg


def validate(test_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(test_loader), [batch_time, losses, top1], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            small_ds = SmallIterator(images, target, 256)
            num_small_batch = len(small_ds)

            output = torch.tensor([])
            loss = torch.tensor(0.0)
            if args.gpu is not None:
                output = output.cuda(args.gpu)
                loss = loss.cuda(args.gpu)

            for small_images, small_target in small_ds:
                # compute output
                small_output = model(small_images)
                output = torch.cat((output, small_output))
                small_loss = criterion(small_output, small_target)
                small_loss /= num_small_batch
                loss += small_loss
                # measure accuracy and record loss

            acc1 = accuracy(output, target)[0]

            # この処理でacc1は平均されて処理されるはず！
            # この処理でlossも平均されて処理されるはず！
            if args.distributed:
                size = float(args.world_size)
                dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
                acc1 /= size
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= size

            # update loss and acc info
            top1.update(acc1[0], images.size(0))
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.rank == 0:
                if i % args.print_freq == 0:
                    progress.display(i)

        if args.rank == 0:
            # TODO: this should also be done with the ProgressMeter
            print(" * Test Acc (avg.) {top1.avg:.3f}".format(top1=top1))

    return top1.avg, losses.avg


# Create random directions for landscape analysis
def create_random_directions(params_ref, dev):
    alpha = []
    beta = []
    for param in params_ref:
        alpha.append(torch.rand(param.shape).to(dev))
        beta.append(torch.rand(param.shape).to(dev))
    return alpha, beta


# get parameters of the model
# WARNING: requires_grad of all clone parameters are set to False
# DO NOT USE FOR TRAINING
def get_params_ref(model):
    w = []
    for param in model.parameters():
        w.append(param.detach().clone())
    return w


# update model parametes
# WARNING: requires_grad of all clone parameters are set to False
# DO NOT USE FOR TRAINING
def update_model_params(model, params_ref, alpha, beta, gamma1, gamma2):
    for param, w, a, b in zip(model.parameters(), params_ref, alpha, beta):
        param.data = w + gamma1 * a + gamma2 * b


def get_landscape(
    base_model,
    test_dl,
    crit,
    dev,
    base_w,
    alpha,
    beta,
    x_range=(-0.5, 0.5),
    y_range=(-0.5, 0.5),
    N=10,
):
    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    X, Y = np.meshgrid(x, y)
    Z = []
    for x_ in x:
        tmp = []
        for y_ in y:
            update_model_params(base_model, base_w, alpha, beta, x_, y_)
            l = validate_model(base_model, test_dl, crit, dev)
            print(l)
            tmp.append(l[0])
        Z.append(tmp)
    Z = np.array(Z).reshape((len(x), len(y)))
    return X, Y, Z


def validate_model(model, test_dl, crit, dev):
    test_loss = 0.0
    total = 0.0
    correct = 0.0

    for x, y in test_dl:
        x = x.to(dev)
        y = y.to(dev)
        si = SmallIterator(x, y)
        si_length = len(si)
        for x_, y_ in si:
            with torch.no_grad():
                o = model(x_)
                l = crit(o, y_)
            l = l / si_length
            test_loss += float(l)
            top1 = torch.argmax(o, axis=1)
            correct += torch.sum(top1 == y_)
        total += len(y)

    test_acc = 100 * correct / total
    test_loss = test_loss / len(test_dl)
    return test_loss, test_acc
