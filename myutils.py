import torch
import torch.nn as nn
import numpy as np
from models import create_model
import math

# Iterator used for pseudo large batch sizes
class SmallIterator:
    def __init__(self, x, y, small_bs=128):
        self.x = x
        self.y = y
        self.sbs = small_bs
        self.curr = 0
        self.total_length = len(x)
        self.length = math.ceil(len(x) / small_bs)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr < self.total_length:
            tmp_x = self.x[self.curr : self.curr + self.sbs]
            tmp_y = self.y[self.curr : self.curr + self.sbs]
            self.curr += self.sbs
            return tmp_x, tmp_y
        else:
            raise StopIteration


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
            l = validate(base_model, test_dl, crit, dev)
            print(l)
            tmp.append(l[0])
        Z.append(tmp)
    Z = np.array(Z).reshape((len(x), len(y)))
    return X, Y, Z


def train(model, train_dl, test_dl, crit, optim, sched, dev, args):
    try:
        import wandb

        wandb.init(project=args.project, name=args.experiment)
        logger = wandb.log
    except:
        logger = print

    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = 0.0
        total = 0.0
        correct = 0.0

        model.train()

        for idx, (x, y) in enumerate(train_dl):
            x = x.to(dev)
            y = y.to(dev)
            optim.zero_grad()
            si = SmallIterator(x, y)
            si_length = len(si)

            if args.optim == "sam":
                # prev_params = get_params_ref(model)
                prev_params = [
                    param.clone().detach().requires_grad_(True)
                    for param in model.parameters()
                ]

            for x_, y_ in si:
                o = model(x_)
                l = crit(o, y_)
                l = l / si_length
                l.backward()
                train_loss += float(l)
                top1 = torch.argmax(o, axis=1)
                correct += torch.sum(top1 == y_)

            if args.optim == "sam":
                optim.step_calc_w_adv()
                optim.zero_grad()

                for x_, y_ in si:
                    o = model(x_)
                    l = crit(o, y_)
                    l = l / si_length
                    l.backward()

                optim.load_original_params_and_step(prev_params)

            else:
                optim.step()

            total += len(y)

        train_loss = train_loss / len(train_dl)
        train_acc = 100.0 * correct / total

        model.eval()
        test_loss, test_acc = validate(model, test_dl, crit, dev)

        sched.step()

        best_acc = max(best_acc, test_acc)

        logger(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        if (
            args.checkpoint != -1
            and args.checkpoint != 0
            and epoch % args.checkpoint == 0
        ):
            torch.save(model.state_dict(), args.path + f"_{epoch}")
            torch.save(test_loss, args.path + f"_loss_{epoch}")

    logger({"best_test_acc": best_acc})

    if args.checkpoint == -1:
        torch.save(model.state_dict(), args.path + "_last")


def validate(model, test_dl, crit, dev):
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


def load_model_from_path(model, path_to_model):
    # state_dict = torch.load(path_to_model)["state"]
    # for param, k in zip(model.parameters(), state_dict):
    #     param.data = state_dict[k]["momentum_buffer"]
    state_dict = torch.load(path_to_model)
    state_dict = [
        state_dict[k] for k in state_dict if "running_" not in k and "tracked" not in k
    ]
    for param, new_param in zip(model.parameters(), state_dict):
        param.data = new_param
