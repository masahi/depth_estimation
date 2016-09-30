import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_loss(file_name):
    data = np.loadtxt(file_name)
    
    iters = data[:, 0].astype(int)
    loss = data[:, 1]
    test_loss = data[:, 2]
    val_loss = data[:, 3]

    plt.gray()    
    plt.figure(figsize=(12,9))
    plt.plot(iters, loss)
    plt.plot(iters, val_loss)
    plt.plot(iters, test_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

def get_nyu_data():
    imgs = np.load('data/nyu/npy/images.npy').transpose((0, 2, 3, 1))
    gts = np.load('data/nyu/npy/depths.npy')
    test_ind = loadmat('data/nyu/mat/splits.mat')['testNdxs'].flatten() - 1
    train_ind = loadmat('data/nyu/mat/splits.mat')['trainNdxs'].flatten() - 1

    return imgs, gts, test_ind, train_ind
    
def plot_depth(pred, imgs, gts, ind):
    plt.gray()    
    for i in range(ind.shape[0]):
        plt.figure(figsize=(18,15))
        plt.subplot(1,3,1)
        plt.imshow(imgs[ind[i]])
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(pred[i])
        plt.axis('off')
    
        plt.subplot(1,3,3)
        plt.imshow(gts[ind[i]])
        plt.axis('off')
        
        plt.show()

def plot_depth_train():
    imgs = np.load('rgb_train.npy')
    gts = np.load('depth_train.npy')
    preds = np.load('pred_train.npy')

    plt.gray()

    print(imgs.shape)
    print(gts.shape)
    print(preds.shape)
    
    for i in range(imgs.shape[0]):
        plt.figure(figsize=(18,15))
        plt.subplot(1,3,1)
        plt.imshow(imgs[i])
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(preds[i])
        plt.axis('off')
    
        plt.subplot(1,3,3)
        plt.imshow(gts[i])
        plt.axis('off')
        
        plt.show()
