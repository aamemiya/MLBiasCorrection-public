#------------------------------------------------
import os
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

#------------------------------------------------
### config

ncdir  = '../DATA_test/coupled_A13/0001/obs_p8_010/LSTM'

#------------------------------------------------

# load nature and observation data
nc = netCDF4.Dataset(ncdir + '/assim.nc','r',format='NETCDF4')
fcst = np.array(nc.variables['vfm'][:], dtype=type(np.float64)).astype(np.float32)
fcst_raw = np.array(nc.variables['vfm_raw'][:], dtype=type(np.float64)).astype(np.float32)
anal = np.array(nc.variables['vam'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

nc = netCDF4.Dataset(ncdir + '/../../nature.nc','r',format='NETCDF4')
nature = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
nc.close

ntime=len(time)

smp_e=1
nplt=200
###
plt.figure()
#plt.scatter(test_labels, test_predictions)
#plt.scatter(fcst[ntime-100:ntime,1], anal[ntime-100:ntime,1]-fcst[ntime-100:ntime,1])
plt.scatter(fcst[ntime-nplt:ntime,smp_e], fcst_raw[ntime-nplt:ntime,smp_e]-nature[ntime-nplt:ntime,smp_e])
#plt.scatter(fcst[ntime-nplt:ntime,smp_e], fcst_raw[ntime-nplt:ntime,smp_e]-fcst[ntime-nplt:ntime,smp_e], c='red')
plt.scatter(fcst[ntime-nplt:ntime,smp_e], fcst[ntime-nplt:ntime,smp_e]-nature[ntime-nplt:ntime,smp_e], c='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('forecast',fontsize=12)
plt.ylabel('forecast - truth',fontsize=12)
#plt.axis('equal')
#plt.axis('square')

plt.ylim(-0.4,0.4)
plt.xlim(-15,20)

plt.tick_params(axis='both', which='major', labelsize=12)

#plt.xlim()
#plt.ylim()
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])
plt.savefig('png/fig7-b.png')
###

