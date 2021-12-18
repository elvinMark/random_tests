import os
import sys

import torch
import torch.nn as nn

from utils import create_parser
from myutils import *
from mydatasets import *
from models import *
from pca import *

import numpy as np
import pickle

parser = create_parser()
args = parser.parse_args()

if args.gpu == None:
    args.gpu = 0

dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

model = create_model(args).to(dev)
load_model_from_path(model, args.model_path)

crit = nn.CrossEntropyLoss()
train_dl, test_dl = create_dataloaders(args)
base_w = get_params_ref(model)

if args.pca is None:
    print("Creating random directions")
    alpha, beta = create_random_directions(base_w, dev)
else:
    print("Loading the pca directions")
    alpha, beta = load_pca_directions(dev)

X, Y, Z = get_landscape(
    model,
    test_dl,
    crit,
    dev,
    base_w,
    alpha,
    beta,
    x_range=args.range,
    y_range=args.range,
)

data_ = {
    "dataset": args.dataset,
    "architecture": args.arch,
    "X": X,
    "Y": Y,
    "Z": Z,
}

with open(f"data_{args.batch_size}_{args.optim}", "wb") as f:
    pickle.dump(data_, f)
    f.close()
