import torch
import torch.nn as nn
from hessian_eigenthings import compute_hessian_eigenthings
from models import create_model
from utils import create_parser
from mydatasets import create_dataloaders
from myutils import load_model_from_path
import pickle

parser = create_parser()
args = parser.parse_args()

bs = args.batch_size
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(args).to(dev)
load_model_from_path(model, args.model_path)

args.batch_size = 128
train_dl, test_dl = create_dataloaders(args)
loss = nn.CrossEntropyLoss()
num_eigenthings = 1

eigenvals, eigenvecs = compute_hessian_eigenthings(
    model, test_dl, loss, num_eigenthings
)

with open(f"eigenvals_{args.lr}_{bs}_{args.optim}_{args.nbs}.pkl", "wb") as f:
    pickle.dump(eigenvals, f)

print(eigenvals)
