import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np
import os
import shutil

ncdir="../DATA/coupled_A13/obs_p8_010/nocorr"

# load nature and observation data
nc = netCDF4.Dataset(ncdir + '/assim.nc','r',format='NETCDF4')
vam = np.array(nc.variables['vam'][:], dtype=type(np.float64)).astype(np.float32)
vfm = np.array(nc.variables['vfm'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

ntime=len(time)

fig,ax = plt.subplots(nrows=2,sharex='all')

element = 0

vmin=-12.0
vmax=24.0
vvmin=-0.6
vvmax= 0.6

length=100

dv=vam-vfm

ax[0].set_ylim(vmin,vmax)
ax[0].plot(time[0:length],vam[-length-1:-1,element], color='black', linewidth=2)
#ax[0].legend(loc='upper right')
ax[0].set_title('(a) Analysis ensemble mean',loc='left')

ax[1].set_ylim(vvmin,vvmax)
ax[1].plot(time[0:length],dv[-length-1:-1,element], color='grey', linewidth=2)
#ax[1].legend(loc='upper right')
ax[1].set_title('(b) Analysis ensemble mean - forecast ensemble mean',loc='left')

ax[0].set_ylabel('X '+str(element+1)+' (a)',fontsize=12)
ax[1].set_ylabel('X '+str(element+1)+' (a-f)',fontsize=12)
ax[1].set_xlabel('time',fontsize=12)

ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)

fig.subplots_adjust(hspace=0.3)

fig.savefig ('png/fig2.png')

