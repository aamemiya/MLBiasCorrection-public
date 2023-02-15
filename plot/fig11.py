#------------------------------------------------
import os
import numpy as np
import netCDF4
#import param
import matplotlib.pyplot as plt

#------------------------------------------------
### config
nx=8

expdir='../DATA/coupled_A13'

obsdir='obs_p8_010'
#obsdir='obs_p6_010'

#dadir=('nocorr','linear','linear4')
dadir=['nocorr','linear','linear4','Dense_single','LSTM']

legends = ['nocorr','linear','4th poly','Dense','LSTM']
#legends = ('nocorr','linear','4th poly')
#lstyles = ('solid','dotted',(0,(1,4)),'dashed','dashdot')
#lstyles = ('solid','dashdot',(0,(3,5,1,5)),(0,(3,2,1,2)))
lstyles = ['solid','dotted','dashdot',(0,(3,5,1,5)),(0,(3,2,1,2))]
#lstyles = ('solid','dotted','dashdot')


#------------------------------------------------

nexp=len(dadir)

rmse=[]
sprd=[]
rmse_plot=[]
sprd_plot=[]


# load nature and observation data
for i in range(nexp):
  nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + dadir[i] + '/stats.nc','r',format='NETCDF4')
  rmse.append(np.array(nc.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32))
  sprd.append(np.array(nc.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32))
  if (i == 0): 
    time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
  nc.close 
  rmse_plot.append(np.mean(rmse[i],axis=0))
  sprd_plot.append(np.mean(sprd[i],axis=0))


ntime=len(time)

doubling_time=0.2*2.1 ### 2.1 day

#refy=np.array((1,max(rmse_plot[0])))
#refx=np.array((0,doubling_time*np.log2(refy[1])))

smp_e=1
nplt=40
###
plt.figure()
plt.yscale('log')
#plt.scatter(test_labels, test_predictions)
#plt.scatter(fcst[ntime-100:ntime,1], anal[ntime-100:ntime,1]-fcst[ntime-100:ntime,1])
for i in range(nexp):
  plt.plot(time[:nplt], rmse_plot[i][:nplt],label=legends[i],linestyle=lstyles[i])
# plt.plot(time, sprd_plot)

plt.legend(bbox_to_anchor=(0.99,0.01), loc='lower right', borderaxespad=0,fontsize=14)
#plt.plot(refx, refy,color='black',linestyle='dashed')
plt.xlabel('time')
plt.ylabel('RMSE')
#plt.axis('equal')
#plt.axis('square')
plt.xlim()
plt.ylim([0.05,8.0])
plt.yticks([0.1,0.2,0.5,1,2,5],['0.1','0.2','0.5','1.0','2.0','5.0'])
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])
plt.savefig('png/fig11.png', dpi = 400, bbox='tight')
###

