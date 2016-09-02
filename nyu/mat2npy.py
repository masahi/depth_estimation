import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os

file_names = open('file_names2.txt').readlines()
fail_list = []
for f in file_names:
    f = f.rstrip()
    print f

    try:
        data = loadmat(f)
        rgb = data['rgb']
        depth = data['depth']
    
        np.save(f + '_rgb.npy', rgb)
        np.save(f + '_depth.npy', depth)
        os.remove(f)
    except:
        fail_list.append(f)
        print f
        
