from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mb
import numpy as np
import pandas as pd

t1=pd.read_csv('trainingData.csv')
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
xs=t1.loc[:,'LONGITUDE']
ys=t1.loc[:,'LATITUDE']
zs=t1.loc[:,'WAP2']
# ax.scatter(xs, ys, zs, c='b', marker='o')
# ax.plot_wireframe(xs, ys, zs, rstride=10,cstride=10)

ax = fig.gca(projection='3d')
# ax.plot_trisurf(xs, ys, zs, linewidth=0.2, antialiased=True)
dx=np.ones_like(xs)
dy=dx.copy()
dz=np.zeros_like(zs)

ax.bar3d(xs,ys,dz,1,1,zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


