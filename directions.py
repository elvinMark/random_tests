from pca import get_pca_directions
import argparse
import torch

parser = argparse.ArgumentParser(
    description="This is a program to calculate the main directions (using pca)"
)

parser.add_argument(
    "--models-list",
    type=str,
    default="models_list.txt",
    help="this is a file that contains the list of models",
)

args = parser.parse_args()
models_list = []

with open(args.models_list, "r") as f:
    for line in f.readlines():
        state_dict = torch.load(line[:-1])
        models_list.append(
            [
                state_dict[k]
                for k in state_dict
                if "running_" not in k and "tracked" not in k
            ]
        )
    f.close()

get_pca_directions(models_list)
