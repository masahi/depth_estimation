import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndimage
from scipy.misc import imread
from skimage import feature

from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=5)
# d = np.load('d.npy')
# h,w = d.shape
# kmeans.fit(d.flatten()[:, np.newaxis])

# labels = kmeans.predict(d.flatten()[:, np.newaxis]).reshape((h, w))
# centers = kmeans.cluster_centers_
# quantized = np.zeros((h, w))

# for i in range(h):
#     for j in range(w):
#         quantized[i, j] = centers[labels[i,j]]

# plt.figure(figsize=(12,9))
# plt.subplot(1,3,1)
# plt.imshow(d)
# plt.subplot(1,3,2)
# plt.imshow(quantized)
# plt.subplot(1,3,3)
# plt.imshow(labels)
# plt.show()
depths = np.load('../data/nyu/npy/depths.npy')
rgbs = np.load('../data/nyu/npy/images.npy')
rgbs = rgbs.transpose((0, 2, 3, 1))
fastms_dir = '/home/masa/research/fastms/output/'
plt.gray()

for i in range(depths.shape[0]):
    plt.figure(figsize=(15,12))
    edge = feature.canny(depths[i], sigma=5)
    fastms_edge = imread(fastms_dir + "%s__result_alpha20_lambda0.1_edges.png" % str(i))
    print i
    
    plt.subplot(1,4,1)
    plt.imshow(depths[i])
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(rgbs[i])
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.imshow(edge)
    plt.axis('off')
    plt.title('canny')

    plt.subplot(1,4,4)
    plt.imshow(fastms_edge)
    plt.axis('off')
    plt.title('MS')

    plt.show()
