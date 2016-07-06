import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

test_ind = loadmat('splits.mat')['testNdxs'] - 1
#test_ind = loadmat('splits.mat')['trainNdxs'] - 1

pred = np.load('pred_m3d.npy')
imgs = np.load('make3d/test_imgs.npy')
gt = np.load('make3d/test_depths.npy')
dt = np.load('make3d/test_depths_dt.npy')

plt.gray()

for i in range(imgs.shape[0]):
#for i in range(1):
    plt.figure(figsize=(18,15))
    plt.subplot(1,4,1)
    plt.imshow(imgs[i][0])
    plt.axis('off')
    
    plt.subplot(1,5,2)
    plt.imshow(pred3[i])
    plt.axis('off')
    plt.title('VGG + upconvolution')

    plt.subplot(1,5,3)
    plt.imshow(pred8[i])
    plt.axis('off')
    plt.title('Without maxpooling')

    plt.subplot(1,5,4)
    plt.imshow(pred9[i])
    plt.axis('off')
    plt.title('Without VGG weight')

    plt.subplot(1,5,5)
    plt.imshow(gts[i][0])
    plt.axis('off')
    plt.title('GT')
    plt.show()

