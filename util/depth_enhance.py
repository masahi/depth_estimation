import os
from scipy.io import loadmat, savemat
from skimage.io import imread

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, spdiags
from pyamg import solve
from depth_to_3d import depth_to_3d

def depth_enhance(depth, edge, smooth_coeff = 10, edge_thres = 0.5):
    rows = []
    cols = []
    rhs = []
    lhs = []

    height, width = depth.shape
    n_pixels = height * width

    for i in range(height):
        for j in range(width):
            index = j + i * width
            e = edge[i,j]

            sum_other_e = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if i + dy < 0 or i + dy >= height:
                        continue
                    if j + dx < 0 or j + dx >= width:
                        continue

                    other_e = edge[i + dy, j + dx]
                    other_index = j + dx + (i + dy) * width
                    rows.append(index)
                    cols.append(other_index)
                    if e > edge_thres or other_e > edge_thres:
                        delta = 0.01
                    else:
                        delta = smooth_coeff

                    lhs.append(-delta)
                    sum_other_e += delta
                    
            rhs.append(depth[i,j])
            rows.append(index)
            cols.append(index)
            lhs.append(1+sum_other_e)

    A = csr_matrix((lhs, (rows, cols)), shape=(n_pixels, n_pixels))
    new_depth = solve(A, np.array(rhs))

    return new_depth.reshape((height, width))


plt.gray()
base_dir = '/home/masa/torch/code/vision/relative_depth/src/experiment/'
file_names = [file_name for file_name in  os.listdir(base_dir + 'imgs') if file_name.endswith('doc.mat')]

for (i, name) in enumerate(file_names):
    if not name.endswith('doc.mat'):
        continue

    imname = name[:-8]
    depth_name = name[:-12] + '_depth.png'
    print(i, imname)
    rgb = imread(base_dir + 'imgs/' + imname)
    edge = loadmat(base_dir + 'imgs/' + name)['edge']
    depth = imread(base_dir + 'imgs/' + depth_name)
    depth_enhanced = depth_enhance(depth, edge)

    #depth_to_3d(rgb, depth, 'mesh/%s.ply' % imname)
    #depth_to_3d(rgb, depth_enhanced, 'mesh/%s_enhanced.ply' % imname)
                    
    plt.figure(figsize=(18,15))        
    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.axis('off')
    plt.subplot(1, 4, 2)    
    plt.imshow(edge > 0.5)
    plt.axis('off')
    plt.title('bounday')
    plt.subplot(1, 4, 3)    
    plt.imshow(depth)
    plt.axis('off')
    plt.title('nips16 depth')
    plt.subplot(1, 4, 4)    
    plt.imshow(depth_enhanced)
    plt.axis('off')
    plt.title('depth edge enhanced')

    plt.show()

    
# file_names = os.listdir(base_dir + 'output_diw')

# for (i, name) in enumerate(file_names):
#     if not name.endswith('doc.mat'):
#         continue

#     imname = name[:-8]
#     depth_name = name[:-16] + '_depth.png'
#     print(imname)
#     rgb = imread(base_dir + 'output_diw/' + imname)
#     edge = loadmat(base_dir + 'output_diw/' + name)['edge']
#     depth = imread(base_dir + 'output_nyu_diw/' + depth_name)
#     depth_enhanced = depth_enhance(depth, edge)

#     depth_to_3d(rgb, depth, 'mesh/%s.ply' % imname)
#     depth_to_3d(rgb, depth_enhanced, 'mesh/%s_enhanced.ply' % imname)
    
#     plt.figure(figsize=(18,15))        
#     plt.subplot(1, 4, 1)
#     plt.imshow(rgb)
#     plt.axis('off')
#     plt.subplot(1, 4, 2)    
#     plt.imshow(edge > 0.5)
#     plt.axis('off')
#     plt.title('bounday')        
#     plt.subplot(1, 4, 3)    
#     plt.imshow(depth)
#     plt.axis('off')
#     plt.title('nips16 depth')        
#     plt.subplot(1, 4, 4)    
#     plt.imshow(depth_enhanced)
#     plt.axis('off')
#     plt.title('depth edge enhanced')
    
#     plt.show()

