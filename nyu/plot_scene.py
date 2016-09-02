import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

plt.gray()

import os

files = open('file_names.txt').readlines()

for f in files:
    f = f.rstrip()
    rgb = np.load(f + '_rgb.npy')
    depth = np.load(f + '_depth.npy')

    from skimage.transform import resize
    rgb = resize(rgb, (104, 144))
    depth = resize(depth, (104, 144))

    print rgb.shape
    
    plt.figure(figsize=(12,9))
    plt.subplot(1,2,1)
    plt.imshow(rgb)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(depth, vmin=0)
    plt.axis('off')

    plt.show()
    
