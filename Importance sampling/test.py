from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.special import erf
a = np.load('fun_vals_1000.npy')
b = a.cumsum() / (np.arange(a.shape[0])+1.)
c = np.zeros_like(a)
c[0] = a[0]
for i in range(1, c.shape[0]):
    c[i] = 0.9*c[i-1] + a[i]
plt.figure()
plt.ylim(-20000, 20000)
plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.show()