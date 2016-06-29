import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

test_ind = loadmat('splits.mat')['testNdxs'] - 1
#test_ind = loadmat('splits.mat')['trainNdxs'] - 1

preds = np.load('pred.npy')
imgs = np.load('images.npy').transpose((0,3,2,1))[test_ind]
gts = np.load('depths.npy').transpose((0,2,1))[test_ind]


plt.gray()

for i in range(imgs.shape[0]):
#for i in range(1):
    plt.figure(figsize=(12,9))
    plt.subplot(1,3,1)
    plt.imshow(imgs[i][0])
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(preds[i])
    plt.axis('off')
    plt.title('Pred')

    plt.subplot(1,3,3)
    plt.imshow(gts[i][0])
    plt.axis('off')
    plt.title('GT')    
    
    plt.show()

