import numpy as np

import os
from scipy.io import loadmat
from pylab import *

test_ind = loadmat('splits.mat')['testNdxs'].flatten() - 1
depths = np.load('depths.npy').transpose((0,2,1))[test_ind]
preds = np.load('pred.npy')

abs_diff = 0
sq_diff = 0
avg_log10 = 0
rmse = 0
rmse_log = 0
n_valid = 0
thres_1 = 0.0
thres_2 = 0.0
thres_3 = 0.0
t = 1.25

for (i,f) in enumerate(test_ind):
    out = preds[i]
    gt = depths[i]
    mask = gt > 0
    out = out[mask]
    gt = gt[mask]

    n_valid += np.sum(mask)
    abs_diff += np.sum(np.abs(out - gt) / gt)
    sq_diff += np.sum(np.abs(out - gt) ** 2 / gt)
    avg_log10 += np.sum(np.abs(np.log10(out) - np.log10(gt)))
    rmse += np.sum((out - gt) ** 2)
    rmse_log += np.sum((np.log(out) - np.log(gt)) ** 2)

    thres_1 += np.sum(np.maximum(out/gt, gt/out) < t)
    thres_2 += np.sum(np.maximum(out/gt, gt/out) < t**2)
    thres_3 += np.sum(np.maximum(out/gt, gt/out) < t**3 )
    
    # print abs_reldiff, rmse
    
    # figure(figsize=(15,12))
    # subplot(1,3,1)
    # imshow(im)
    # axis('off')
    # subplot(1,3,2)
    # imshow(out)
    # title('Prediction')
    # axis('off')    
    # subplot(1,3,3)
    # imshow(depths[:,:,f])
    # axis('off')
    # title('Ground Truth')
    # show()

abs_diff /= n_valid    
sq_diff /= n_valid    
avg_log10 /= n_valid    
rmse = np.sqrt(rmse/n_valid)
rmse_log = np.sqrt(rmse_log/n_valid)
thres_1 /= n_valid
thres_2 /= n_valid
thres_3 /= n_valid

print abs_diff, sq_diff, avg_log10, rmse, rmse_log, thres_1, thres_2, thres_3
