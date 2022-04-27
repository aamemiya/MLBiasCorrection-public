import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import os
import shutil

ncdir="../DATA/coupled_A13/obs_p6_010"

# load nature and observation data
nc_obs = netCDF4.Dataset(ncdir + '/obs.nc','r',format='NETCDF4')
vobs = np.array(nc_obs.variables['vy'][:], dtype=type(np.float64)).astype(np.float32)
nc_obs.close 

nc = netCDF4.Dataset(ncdir + '/nocorr/assim.nc','r',format='NETCDF4')
vam = np.array(nc.variables['vam'][:], dtype=type(np.float64)).astype(np.float32)
va = np.array(nc.variables['va'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

ntime=len(time)

timeh=time-0.5*(time[1]-time[0])

vas=np.std(va,axis=1)
nx=vam.shape[1]

fig,ax = plt.subplots(nrows=3,sharex='all')

divider = [make_axes_locatable(a) for a in ax]

cax = dict()
for i in range(3):
	cax [i] = divider[i].append_axes('right', size='5%', pad=0.05)

delta= dict()

element = 0

vobs[vobs==-9.99e8] = np.nan


#for i in range(10) : 
#  print(vobs[i-1,:])
#quit()

vmin=-12.0
vmax=24.0
vmins=0.0
vmaxs=1.0


length=20
stepe=29999
steps=stepe-length

ax[0].set_ylim(0.5,float(nx)+0.5)
ax[1].set_ylim(0.5,float(nx)+0.5)
ax[2].set_ylim(0.5,float(nx)+0.5)
delta[0]=ax[0].imshow(vobs.T[:,steps:stepe],cmap=plt.get_cmap('viridis'),vmin=vmin,vmax =vmax,extent=[timeh[0],timeh[length],0.5,float(nx)+0.5],aspect='auto')
delta[1]=ax[1].imshow(vam.T[:,steps:stepe],cmap=plt.get_cmap('viridis'),vmin=vmin,vmax =vmax,extent=[timeh[0],timeh[length],0.5,float(nx)+0.5],aspect='auto')
delta[2]=ax[2].imshow(vas.T[:,steps:stepe],cmap=plt.get_cmap('YlOrRd'),vmin=vmins,vmax =vmaxs,extent=[timeh[0],timeh[length],0.5,float(nx)+0.5],aspect='auto')

ax[0].set_title('(a) Observation',loc='left')
ax[1].set_title('(b) Analysis mean',loc='left')
ax[2].set_title('(c) Analysis spread',loc='left')

ax[0].set_ylabel('x',fontsize=12)
ax[1].set_ylabel('x',fontsize=12)
ax[2].set_ylabel('x',fontsize=12)
ax[2].set_xlabel('time',fontsize=12)

ax[0].set_yticks([1,nx])
ax[1].set_yticks([1,nx])
ax[2].set_yticks([1,nx])

for i in range(3):
	fig.colorbar(delta[i],cax=cax[i],orientation='vertical')

ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[2].tick_params(axis='both', which='major', labelsize=12)


fig.subplots_adjust(hspace=0.4)

fig.savefig ('png/fig3.png')


