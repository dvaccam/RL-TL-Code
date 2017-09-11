from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
epsp = 20*7.
epsv = 0.14*7.
xs = np.linspace(-11, 11, 100)
ys = np.linspace(-0.8, 0.8, 100)
xs, ys = np.meshgrid(xs, ys)
xs = xs.flatten()
ys = ys.flatten()
p1 = np.array([-10])
p2 = np.array([10])
p3 = np.array([0])
v1 = np.array([-0.7])
v2 = np.array([0.7])
v3 = np.array([0])
dp1 = np.abs(xs - p1)
dp2 = np.abs(xs - p2)
dp3 = np.abs(xs - p3)
dv1 = np.abs(ys - v1)
dv2 = np.abs(ys - v2)
dv3 = np.abs(ys - v3)
d = np.stack((dp1**2/epsp + dv1**2/epsv + 1, dp1**2/epsp + dv2**2/epsv + 1,
              dp2**2/epsp + dv1**2/epsv + 1, dp2**2/epsp + dv2**2/epsv + 1,
              dp3**2/epsp + dv3**2/(2*epsv))).T
zs = (np.hstack((1./(1.+d[:,:-1]), (d[:,-1]/(1.+d[:,-1])).reshape((-1,1))))*np.array([50, 50, 80., 80., 0])).sum(axis=1)
d = np.stack((np.exp(-dp1**2/epsp - dv1**2/epsv), np.exp(-dp1**2/epsp - dv2**2/epsv),
              np.exp(-dp2 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp2**2/epsp - dv2**2/epsv),
              dp3**2/epsp + dv3**2/(2*epsv))).T
zs = (np.hstack((d[:,:-1], (d[:,-1]/(1.+d[:,-1])).reshape((-1,1))))*np.array([50, 50, 60., 60., 0])).sum(axis=1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_surface(xs.reshape((100,100)), ys.reshape((100,100)), zs.reshape((100,100)))
d = np.stack((dp1**2/epsp + dv1**2/epsv + 1, dp1**2/epsp + dv2**2/epsv + 1,
              dp2**2/epsp + dv1**2/epsv + 1, dp2**2/epsp + dv2**2/epsv + 1)).T
zs = (1./(1.+d)).sum(axis=1)
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
