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

test_loss, test_acc = validate(model, test_dl, crit, dev)

print(test_loss, test_acc)
