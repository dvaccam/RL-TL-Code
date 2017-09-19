from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
epsp = (20*0.3)**2
epsv = (1.4*.3)**2
xs = np.linspace(-11, 11, 100)
ys = np.linspace(-0.8, 0.8, 100)
as_ = np.linspace(-0.9, 0.9, 10)
xs, ys = np.meshgrid(xs, ys)
xs = xs.flatten()
ys = ys.flatten()


p1 = np.array([-10])
p3 = np.array([10])
p2 = np.array([0])
v1 = np.array([-0.7])
v3 = np.array([0.7])
v2 = np.array([0])
dp1 = np.abs(xs - p1)
dp2 = np.abs(xs - p2)
dp3 = np.abs(xs - p3)
dv1 = np.abs(ys - v1)
dv2 = np.abs(ys - v2)
dv3 = np.abs(ys - v3)
d = np.stack((np.exp(-dp1**2/epsp - dv1**2/epsv), np.exp(-dp1**2/epsp - dv2**2/epsv), np.exp(-dp1**2/epsp - dv3**2/epsv),
              np.exp(-dp2 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp2**2/epsp - dv2**2/epsv), np.exp(-dp2**2/epsp - dv3**2/epsv),
              np.exp(-dp3 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp3 ** 2 / epsp - dv2 ** 2 / epsv),np.exp(-dp3 ** 2 / epsp - dv3 ** 2 / epsv),
              np.ones(dp1.shape[0], dtype=np.float64))).T
zs = (d*np.array([60., 65., 70., 50., 0., 80., 40., 65., 90., 0.])).sum(axis=1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_surface(xs.reshape((100,100)), ys.reshape((100,100)), zs.reshape((100,100)))
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
