import numpy as np
from sklearn.decomposition import PCA
import pickle
import torch

# Get 2 main pca directions of the models parameters
def get_pca_directions(models):
    dir1 = []
    dir2 = []
    N = len(models[0])
    for i in range(N):
        tmp_ = []
        shape_ = models[0][i].shape
        for model in models:
            model_ = model[i].detach().clone().cpu().numpy().reshape(-1)
            tmp_.append(model_)
        tmp_ = np.array(tmp_)
        pca = PCA(n_components=2)
        pca.fit(tmp_)
        dir1.append(pca.components_[0].reshape(shape_))
        dir2.append(pca.components_[1].reshape(shape_))
    with open("directions/direction1.pkl", "wb") as f:
        pickle.dump(dir1, f)
        f.close()
    with open("directions/direction2.pkl", "wb") as f:
        pickle.dump(dir2, f)
        f.close()


def load_pca_directions(dev):
    with open("directions/direction1.pkl", "rb") as f:
        dir1 = pickle.load(f)
        f.close()
    with open("directions/direction2.pkl", "rb") as f:
        dir2 = pickle.load(f)
        f.close()
    dir1 = [torch.from_numpy(w_).float().to(dev) for w_ in dir1]
    dir2 = [torch.from_numpy(w_).float().to(dev) for w_ in dir2]
    return dir1, dir2
