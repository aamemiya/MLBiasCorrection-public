import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np
import os

ncdir="../DATA/coupled_A13"

# load nature and observation data
nc = netCDF4.Dataset(ncdir + '/nature_full_fast.nc','r',format='NETCDF4')
v = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
vv = np.array(nc.variables['vv'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

ntime=len(time)


fig,ax = plt.subplots(nrows=2,sharex='all')

element = 0
elementy = 16


vmin=-10.0
vmax= 20.0
vvmin=-1.5
vvmax= 1.5

length=1000


ax[0].set_ylim(vmin,vmax)
ax[0].plot(time[0:length],v[-length-1:-1,element], color='black', linewidth=2)
#ax[0].legend(loc='upper right')
ax[0].set_title('(a) Slow variable',loc='left')

ax[1].set_ylim(vvmin,vvmax)
ax[1].plot(time[0:length],vv[-length-1:-1,elementy], color='grey', linewidth=2)
#ax[1].legend(loc='upper right')
ax[1].set_title('(b) Fast variable',loc='left')

ax[0].set_ylabel('X '+str(element+1), fontsize=12)
ax[1].set_ylabel('Y '+str(element+1), fontsize=12)
ax[1].set_xlabel('time', fontsize=12)

ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)

fig.subplots_adjust(hspace=0.3)

fig.savefig ('png/fig1.png')

