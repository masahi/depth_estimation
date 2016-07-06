import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#test_ind = loadmat('splits.mat')['trainNdxs'] - 1

pred = np.load('pred_m3d.npy')
imgs = np.load('make3d/test_imgs.npy').transpose((2,3,1,0))
gt = np.load('make3d/test_depths.npy')
dt = np.load('make3d/test_depths_dt.npy')

plt.gray()

for i in range(imgs.shape[0]):
#for i in range(1):
    plt.figure(figsize=(15,12))
    plt.subplot(1,4,1)
    plt.imshow(imgs[:,:,:,i])
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.imshow(pred[i])
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(dt[:,:,i])
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(gt[:,:,i])
    plt.axis('off')
    
    plt.show()

