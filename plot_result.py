import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

test_ind = loadmat('splits.mat')['testNdxs'] - 1
#test_ind = loadmat('splits.mat')['trainNdxs'] - 1

pred3 = np.load('pred3.npy')
pred8 = np.load('pred8.npy')
pred9 = np.load('pred9.npy')
imgs = np.load('images.npy').transpose((0,3,2,1))[test_ind]
gts = np.load('depths.npy').transpose((0,2,1))[test_ind]

plt.gray()

for i in range(imgs.shape[0]):
#for i in range(1):
    plt.figure(figsize=(18,15))
    plt.subplot(1,5,1)
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

