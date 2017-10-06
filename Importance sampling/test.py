from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.special import erf
epsp = (20*0.3)**2
epsv = (1.4*.3)**2
xs = np.linspace(0., 1., 100)
ys = np.linspace(0., 1., 100)
xs, ys = np.meshgrid(xs, ys)
fig = plt.figure()
zs = np.abs((erf((0.6 - xs*0.7) / (0.2 * np.sqrt(2.))) - erf((0.4 - xs*0.7) / (0.2 * np.sqrt(2.))))/2. - (erf((0.6 - ys*0.7) / (0.2 * np.sqrt(2.))) - erf((0.4 - ys*0.7) / (0.2 * np.sqrt(2.))))/2.)
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_surface(xs, ys, zs)
zs1 = 2*0.7*np.abs(xs-ys)/(np.sqrt(2*np.pi)*0.2)
sur = ax.plot_surface(xs, ys, zs1)
plt.show()
plt.close()







epsp = (20*.1)**2
epsv = (1.4*.1)**2
epsa = (2.*.1)**2
ps = np.linspace(-10, 10, 4)
vs = np.linspace(-0.7, 0.7, 4)
acts = np.linspace(-1, 1, 4)
idx_cube = np.stack(np.meshgrid(np.arange(ps.shape[0]), np.arange(vs.shape[0]), np.arange(acts.shape[0]), indexing='ij'))
idx_cube = np.transpose(idx_cube, axes=(1, 2, 3, 0)).reshape((-1,3))
dists_p = (xs.reshape((-1,1)) - ps.reshape((1,-1)))/20.
dists_v = (ys.reshape((-1,1)) - vs.reshape((1,-1)))/1.4
for i in range(1):
    dists_a = (np.repeat(as_[i], xs.shape[0]).reshape((-1, 1)) - acts.reshape((1, -1))) / 2.
    dists = dists_p[:,idx_cube[:,0]]**2 + dists_v[:,idx_cube[:,1]]**2 + dists_a[:,idx_cube[:,2]]**2
    zs = np.hstack((np.exp(-dists/(0.15**2)), np.ones((dists.shape[0], 1), dtype=np.float64))).sum(axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sur = ax.plot_surface(xs.reshape((100,100)), ys.reshape((100,100)), zs.reshape((100,100)))
plt.show()
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_surface(xs.reshape((100,100)), ys.reshape((100,100)), zs.reshape((100,100)))
d = dp3**2/epsp + dv3**2/(2*epsv)
zs = (d/(1.+d))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_surface(xs.reshape((100,100)), ys.reshape((100,100)), zs.reshape((100,100)))'''

plt.show()
plt.close()
