# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
# from vpython import *
# scene.width = 1200
# scene.height = 900
# L_xyz= 360
# _R=0.6
# _d = L_xyz- 2
# xaxis = cylinder(pos=vec(20, 5, 0), axis=vec(_d, 0, 0), radius=_R, color=color.yellow)
# yaxis = cylinder(pos=vec(20, 5, 0), axis=vec(0, _d, 0), radius=_R, color=color.yellow)
# zaxis = cylinder(pos=vec(20, 5, 0), axis=vec(0, 0, _d-100), radius=_R, color=color.yellow)
# _h = 0.05 * L_xyz
# text(pos=xaxis.pos + 1.02 * xaxis.axis, text='x', height=_h, align='center', billboard=True, emissive=True)
# text(pos=yaxis.pos + 1.02* yaxis.axis, text='y', height=_h, align='center', billboard=True, emissive=True)
# text(pos=zaxis.pos + 1.02 * zaxis.axis, text='z', height=_h, align='center', billboard=True, emissive=True)
#
# scene.center=vec(80,110,0)

def normalizeX(arr):

    res = np.copy(arr).astype(np.float)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if res[i][j] == 100:
                res[i][j] = 0
            else:
                res[i][j] = 0.01 * (100+res[i][j])
    return res
t1=pd.read_csv('trainingData.csv')
rows,cols=np.shape(t1)
long=t1.loc[:,'LONGITUDE'].values.reshape(rows,1)
lati=t1.loc[:,'LATITUDE'].values.reshape(rows,1)
floor=t1.loc[:,'FLOOR'].values.reshape(rows,1)
building=t1.loc[:,'BUILDINGID'].values.reshape(rows,1)

x_min=np.min(long)
x_max=np.max(long)
x_length=x_max-x_min

scaled_long=(long-x_min)

y_min=np.min(lati)
y_max=np.max(lati)
y_length=y_max-y_min

scaled_lati=(lati-y_min)

class pot():
    def __init__(self,wap_num,floor_n=None,building_n=None):
        rssi = t1.loc[:,wap_num].values.reshape(-1, 1)
        self.xs = scaled_long
        self.ys = scaled_lati
        if (floor_n!=None)&(building_n!=None):
            rssi=rssi[(t1.loc[:,'FLOOR']==floor_n)&(t1.loc[:,'BUILDINGID']==building_n)]
            self.xs=self.xs[(t1.loc[:,'FLOOR']==floor_n)&(t1.loc[:, 'BUILDINGID'] == building_n)]
            self.ys=self.ys[(t1.loc[:,'FLOOR']==floor_n)&(t1.loc[:, 'BUILDINGID'] == building_n)]
            xx=np.reshape(np.hstack((self.xs, self.ys)), (-1, 2))
        # rssi=normalizeX(rssi)
        self.grid_x, self.grid_y = np.mgrid[np.min(self.xs):np.max(self.xs):200j,np.min(self.ys):np.max(self.ys):200j]

        # print(grid_xy[:3])
        self.zs =rssi
        self.f=floor_n
        self.b=building_n

        poi = np.array([[xs,ys] for xs,ys in np.hstack((self.xs,self.ys))])

        self.grid_z1 = griddata(poi, self.zs, (self.grid_x, self.grid_y), method='linear')
        # self.grid_z1_hull=griddata(poi, self.zs, (self.grid_points_hull[:,0], self.grid_points_hull[:,1]), method='linear')

    def get_in_hull(self,points):
        hull_points_original=self.hull_points
        # print(hull_points_original)
        hull=ConvexHull(hull_points_original).vertices
        # print(hull)
        in_hull_points=[]
        for p in points:
            new_points=list(hull_points_original)+list([p])
            now_hull=ConvexHull(new_points).vertices
            print(now_hull)
            if len(now_hull)==len(hull):
                if(now_hull.any()-hull.any())==0:
                    in_hull_points.append(p)
        return in_hull_points

#原始数据
# point = pot(wap_num='WAP011',floor_n=None)
# posit=[vec(x,y,z*50+f*50) for x,y,z,f in np.hstack((point.xs,point.ys,point.zs,floor))]
# for i in range(len(posit)):
#     points(pos=posit[i],radius=1.5,color=vec(1-point.zs[i],0.5,point.zs[i]))

#扩增数据
def expand_data_ap(AP,n_floor,n_build):
    rssi_arr=[]
    pos_long=[]
    pos_x=[]
    pos_y=[]
    point = pot(wap_num=AP, floor_n=n_floor,building_n=n_build)
    for ix,x in enumerate(np.linspace(np.min(point.xs),np.max(point.xs),200)):
        for iy,y in enumerate(np.linspace(np.min(point.ys),np.max(point.ys),200)):
            if np.isnan(point.grid_z1[ix][iy]):
                continue
            else:
                rssi_arr.append(point.grid_z1[ix][iy])
                pos_x.append(x)
                pos_y.append(y)
    rssi=pd.Series(np.reshape(rssi_arr,(-1)))
    position_x=pd.Series(pos_x)
    position_y=pd.Series(pos_y)
    position_x=position_x+x_min
    position_y=position_y+y_min
    return rssi,position_x,position_y
    # rssi.to_csv(AP+'.csv', index=False, sep=',')
                # points(pos=vec(x,y, (point.grid_z1[ix][iy])*50.0+point.f*50), radius=1.5,color=vec(1-point.grid_z1[ix][iy], 0.5, point.grid_z1[ix][iy]))
colu=['WAP'+str(a)+str(b)+str(c) for a in range(6) for b in range(10) for c in range (10) ]
colu=colu[1:521]
colu+=['LONGITUDE','LATITUDE','FLOOR','BUILDINGID']

all_data=[]
for b in range(3):
    for f in range(5):
        Data = pd.DataFrame(data=None, columns=colu)
        if (b!=2)&(f==4):
            break
        for ap in range(1,521):
            if len(str(ap))==1:
                ap='WAP00'+str(ap)
            elif len(str(ap))==2:
                ap='WAP0'+str(ap)
            else:
                ap='WAP'+str(ap)
            df_rssi,long,lati=expand_data_ap(ap,f,b)
            Data.loc[:,'LONGITUDE']=long
            Data.loc[:,'LATITUDE']=lati
            Data.loc[:, ap] =df_rssi
            Data.loc[:,'FLOOR']=f
            Data.loc[:,'BUILDINGID']=b
        # all_data.append(Data)
        Data.to_csv('train_b'+str(b)+'f'+str(f)+'.csv', index=False,sep=',')
# D_all=pd.concat(all_data)
# D_all.to_csv('new_all.csv',index=False,sep=',')