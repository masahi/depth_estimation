import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

scene_name = 'basement_0001a'

plt.gray()

import os

files = sorted(os.listdir('training/%s' % scene_name))

for f in files:
    data = loadmat('training/%s/%s' % (scene_name,f))
    rgb = data['rgb']
    depth = data['depth']
    
    plt.figure(figsize=(12,9))
    plt.subplot(1,2,1)
    plt.imshow(rgb)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(depth)
    plt.axis('off')
    
    plt.show()
    
