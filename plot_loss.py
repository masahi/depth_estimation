import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = np.loadtxt('loss9.log')

iters = data[:, 0].astype(int)
loss = data[:, 1]

plt.figure(figsize=(12,9))
plt.plot(iters, loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('fig.pdf')
plt.show()


