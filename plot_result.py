import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

test_ind = loadmat('splits.mat')['testNdxs'].flatten() - 1
train_ind = loadmat('splits.mat')['trainNdxs'].flatten() - 1

pred = np.load('pred.npy')
imgs = np.load('images.npy').transpose((0, 2, 3, 1))
gts = np.load('depths.npy')

plt.gray()

def plot_depth(ind):
    for i in range(ind.shape[0]):
        plt.figure(figsize=(18,15))
        plt.subplot(1,3,1)
        plt.imshow(imgs[ind[i]])
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(pred[ind[i]])
        plt.axis('off')
    
        plt.subplot(1,3,3)
        plt.imshow(gts[ind[i]])
        plt.axis('off')
        
        plt.show()
