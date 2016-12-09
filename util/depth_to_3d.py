import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from skimage.io import imread

def depth_to_3d(rgb, depth, fname):

    h, w = depth.shape
    coords = np.zeros((h*w, 3))
    colors = np.zeros((h*w, 3), dtype = np.ubyte)
    offset = 10
    count = 0
    for i in range(-h/2,h/2):
        for j in range(-w/2, w/2):
            
            x = float(j) / (w/2)
            y = float(i) / (h/2)
            z = 1

            ray = np.array([x,y,z])
            norm = np.linalg.norm(ray)

            im_y = i + h/2
            im_x = j + w/2            
            coord_3d = ray * (depth[im_y, im_x] + offset + 1)
            color = rgb[im_y, im_x]

            coords[count] = coord_3d
            colors[count] = color
            count += 1

    n_point = h*w
    header = "ply\n format ascii 1.0\n comment Kinect v1 generated\n element vertex %d\n property double x\n property double y\n property double z\n property uchar red\n property uchar green\n property uchar blue\n end_header\n" % (n_point)
    with open(fname, 'w') as f:
        f.write(header)
        f.write('\n')
        for i in range(n_point):

            for j in range(3):
                f.write(str(coords[i,j]))
                f.write(' ')

            for j in range(3):
                f.write(str(colors[i,j]))
                f.write(' ')

            f.write('\n')

    return coords

def gen_ply():
    file_names = open('file_names.txt').readlines()
    for (i, name) in enumerate(file_names):
        rgb = imread('imgs/' + name.rstrip())
        
        plt.figure(figsize=(12,9))        
        depth = np.load('result/' + str(i+1) + '.npy')
    
        print name
        depth_to_3d(rgb, depth, 'ply/' + name.rstrip()[:-4] + '.ply')
