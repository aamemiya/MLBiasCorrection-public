#------------------------------------------------
import os
import numpy as np
import netCDF4
#import param
import matplotlib.pyplot as plt

#------------------------------------------------
### config
#nx  = param.param_model['dimension'] 
nx=8
#expdir = param.param_exp['expdir']
#obsdir = param.param_exp['obs_type']
#dadir  = param.param_exp['da_type']

expdir='../DATA/coupled_A13'

obsdir='obs_p8_010'
#obsdir='obs_p6_010'

dadir=('truth','nocorr','linear','linear4','dense_single','LSTM')
#dadir=('truth','nocorr','linear','linear4')
#dadir=('nocorr','linear_offline','D_tanh_3','L2D_tanh_3','D_tanh_5','L2D_tanh_5')
#dadir=('nocorr','CLD_tanh5_5','CLD_tanh7_5','CLD_tanh10_5')

legends = ('truth','nocorr','linear','linear 4th','Dense_single','LSTM')
#legends = ('truth','nocorr','linear','linear 4th')
#legends = ('nocorr','linear','Dense-5','LSTM-5','Dense-3','LSTM-3')
#lstyles = ('solid','dotted',(0,(1,4)),'dashed','dashdot')
#lstyles = ('solid','dashdot',(0,(3,5,1,5)),(0,(3,2,1,2)))
lstyles = ('solid','solid','dotted','dashdot',(0,(3,5,1,5)),(0,(3,2,1,2)))
#lstyles = ('solid','solid','dotted','dashdot')


#------------------------------------------------

nexp=len(dadir)

rmse=[]
sprd=[]
rmse_plot=[]
sprd_plot=[]


# load nature and observation data
i=0
nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + 'fcst_from_nature' + '/stats_fullmodel.nc','r',format='NETCDF4')
rmse.append(np.array(nc.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32))
sprd.append(np.array(nc.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32))
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 
rmse_plot.append(np.mean(rmse[i],axis=0))
sprd_plot.append(np.mean(sprd[i],axis=0))


for i in range(1,nexp):
  nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + 'fcst_from_nature' + '/stats_'+ dadir[i] + '.nc','r',format='NETCDF4')
  rmse.append(np.array(nc.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32))
  sprd.append(np.array(nc.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32))
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

  if i == 0 : 
    plt.plot(time[:nplt], rmse_plot[i][:nplt],label=legends[i],linestyle=lstyles[i],color='black')
  else :
    plt.plot(time[:nplt], rmse_plot[i][:nplt],label=legends[i],linestyle=lstyles[i])
# plt.plot(time, sprd_plot)

plt.legend(bbox_to_anchor=(0.99,0.01), loc='lower right', borderaxespad=0,fontsize=14)
#plt.plot(refx, refy,color='black',linestyle='dashed')
plt.xlabel('time')
plt.ylabel('RMSE')
#plt.axis('equal')
#plt.axis('square')
plt.xlim()
plt.ylim([0.01,8.0])
plt.yticks([0.02,0.05,0.1,0.2,0.5,1,2,5],['0.02','0.05','0.1','0.2','0.5','1.0','2.0','5.0'])
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])
plt.savefig('png/fig14.png', dpi = 400, bbox='tight')
###

