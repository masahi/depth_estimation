import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

pred = np.load('pred_train.npy')
imgs = np.load('rgb_train.npy')
gts = np.load('depth_train.npy')

plt.gray()

for i in range(pred.shape[0]):
    plt.figure(figsize=(18,15))
    plt.subplot(1,3,1)
    plt.imshow(imgs[i])
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.imshow(pred[i])
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(gts[i])
    plt.axis('off')
    
    plt.show()
