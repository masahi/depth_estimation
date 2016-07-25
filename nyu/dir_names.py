import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

train_ind = loadmat('../splits.mat')['trainNdxs'].flatten() - 1
scenes = loadmat('../scenes.mat')['scenes']

train_scene = np.unique(scenes[train_ind])

scene_dirs = []
import glob

all_names = [name[5:] for name in glob.glob('data/*')]

for scene in train_scene:
    name = scene[0]

    names = glob.glob('data/%s*' % name)

    for n in names:
        scene_dirs.append(n[5:])

#savemat('train_scenes.mat', {'train_scenes':scene_dirs})
not_used = list(set(all_names) - set(scene_dirs))
