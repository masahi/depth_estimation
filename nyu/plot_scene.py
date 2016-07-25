import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

plt.gray()

import os

files = open('file_names_shuffled.txt').readlines()

for f in files:
    f = f.rstrip()
    data = loadmat(f)
    rgb = data['rgb']
    depth = data['depth']
    depth_filled = data['depth_filled']

    from skimage.transform import resize
    rgb = resize(rgb, (104, 144))
    depth = resize(depth, (104, 144))
    depth_filled = resize(depth_filled, (104, 144))

    print rgb.shape
    
    plt.figure(figsize=(12,9))
    plt.subplot(1,3,1)
    plt.imshow(rgb)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(depth, vmin=0)
    plt.axis('off')
    

    plt.subplot(1,3,3)
    plt.imshow(depth_filled, vmin=0)
    plt.axis('off')

    plt.show()
    
