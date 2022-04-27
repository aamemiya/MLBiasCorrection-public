#------------------------------------------------
import os
import numpy as np
import numpy.linalg as lin
import netCDF4
import matplotlib.pyplot as plt

#------------------------------------------------
### config

ncdir_obs = '../DATA/coupled_A13/obs_p6_001'
ncdir_in  = '../DATA/coupled_A13/obs_p6_001/nocorr'
ncdir_out  = ncdir_in 

reg=0.0  ### regularization

num_order = 1
num_loc_grid = 5 

istart = 100 

#------------------------------------------------

# load nature and observation data
nc = netCDF4.Dataset(ncdir_in + '/assim.nc','r',format='NETCDF4')
ncobs = netCDF4.Dataset(ncdir_obs + '/obs.nc','r',format='NETCDF4')
fcst = np.array(nc.variables['vfm'][:], dtype=type(np.float64)).astype(np.float32)
fcst_raw = np.array(nc.variables['vfm_raw'][:], dtype=type(np.float64)).astype(np.float32)
anal = np.array(nc.variables['vam'][:], dtype=type(np.float64)).astype(np.float32)
anal_mem = np.array(nc.variables['va'][:], dtype=type(np.float64)).astype(np.float32)
obs = np.array(ncobs.variables['vy'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

ntime=len(time)
nx=fcst.shape[1]

anal_sprd=np.std(anal_mem,axis=1)
anal_sprd_copy=np.copy(np.std(anal_mem,axis=1))

###
###  Simple linear regression with fixed basis functions
###
def custom_basisf(x_in) :
    nsmp=x_in.shape[0]
    nx_loc=x_in.shape[1]
    p_out=np.zeros([nsmp,1+nx_loc*num_order])
    for ismp in range(nsmp) :   
      pele=np.ones(1) ### steady component
      for order in range(num_order) :
        pele=np.concatenate([pele,x_in[ismp]**(order+1)])  
      p_out[ismp]=pele
    return p_out


fcst_new_smp=np.empty([0,num_loc_grid])
anal_new_smp=np.empty([0,1])
sprd_new_smp=np.empty([0,1])
for ix in range(nx) : 
  print(ix,'...') 
  indx=np.arange(ix-int(num_loc_grid/2),ix+int(num_loc_grid/2)+1) % nx 
  for it in range(istart,ntime) :
#      if (round(it/100,3).is_integer()):
#      if (obs[it,ix] > -1.0e2 ) : 
        fcst_new_smp=np.concatenate((fcst_new_smp,fcst[np.newaxis,it,list(indx)]),axis=0)
        anal_new_smp=np.concatenate((anal_new_smp,anal[np.newaxis,it,[ix]]-fcst[np.newaxis,it,[ix]]),axis=0)
#        sprd_new_smp=np.concatenate((sprd_new_smp,anal_sprd[np.newaxis,it,[ix]]),axis=0)
        if (obs[it,ix] > -1.0e2 ) : 
#          print('   obs',anal_sprd[it,ix])
          anal_sprd_copy[it,ix]=anal_sprd_copy[it,ix]*5.0
          sprd_new_smp=np.concatenate((sprd_new_smp,anal_sprd_copy[np.newaxis,it,[ix]]),axis=0)
        else : 
#          print('no obs',anal_sprd[it,ix])
          sprd_new_smp=np.concatenate((sprd_new_smp,anal_sprd_copy[np.newaxis,it,[ix]]),axis=0)

#  anal_new_smp=np.concatenate((anal_new_smp,anal[istart:,[ix]]),axis=0)

p_fcst=custom_basisf(fcst_new_smp)


#####
##### 
#####

#arrayb=np.matmul(np.transpose(p_fcst),anal_new_smp)
#arraya=np.matmul(np.transpose(p_fcst),p_fcst)
#arraya=arraya+reg*np.ones(arraya.shape)

p_fcst_fact=np.copy(p_fcst)

for i in range(p_fcst_fact.shape[0]) : 
  p_fcst_fact[i] = p_fcst_fact[i] / sprd_new_smp[i,0]**2

arrayb=np.matmul(np.transpose(p_fcst_fact),anal_new_smp)
arraya=np.matmul(np.transpose(p_fcst_fact),p_fcst)
arraya=arraya+reg*np.ones(arraya.shape) / np.average(sprd_new_smp[:,0]**2)

#####
#####
#####


arrayw=lin.solve(arraya,arrayb)

for ix in range(nx) : 
  indx=np.arange(ix-int(num_loc_grid/2),ix+int(num_loc_grid/2)+1) % nx 
#  fcst[:,[ix]]=np.matmul(custom_basisf(fcst[:,list(indx)]),arrayw)
  fcst[:,[ix]] += np.matmul(custom_basisf(fcst[:,list(indx)]),arrayw)



train_rmse_raw=np.sqrt(np.average((anal_new_smp)**2))
train_rmse=np.sqrt(np.average((np.matmul(p_fcst,arrayw)-anal_new_smp)**2))

print('train RMSE raw',train_rmse_raw)
print('train RMSE fix',train_rmse)


### monitor RMSE

rmse_raw=np.sqrt(np.average((fcst_raw[istart:]-anal[istart:])**2))
rmse=np.sqrt(np.average((fcst[istart:]-anal[istart:])**2))

#bias_raw=np.average(fcst_raw-anal,0)
#bias=np.average(fcst-anal,0)
#rmse_raw=np.average(((fcst_raw-anal)-np.tile(bias_raw,(fcst.shape[0],1)))**2)
#rmse=np.average(((fcst-anal)-np.tile(bias,(fcst.shape[0],1)))**2)

print('RMSE raw',rmse_raw)
print('RMSE fix',rmse)

### Output
###

nc = netCDF4.Dataset(ncdir_out + '/assim_out.nc','w',format='NETCDF3_CLASSIC')
nc.createDimension('x',nx)
nc.createDimension('t',None)
x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
vam_in = nc.createVariable('vam',np.dtype('float64').char,('t','x'))
vfm_in = nc.createVariable('vfm',np.dtype('float64').char,('t','x'))
vfm_raw_in = nc.createVariable('vfm_raw',np.dtype('float64').char,('t','x'))
x_in[:] = np.array(range(1,1+nx))
t_in[:] = np.round(time,4)
vam_in[:,:] = anal
vfm_in[:,:] = fcst
vfm_raw_in[:,:] = fcst_raw
nc.close 


nc = netCDF4.Dataset(ncdir_out + '/coeffw.nc','w',format='NETCDF3_CLASSIC')
nc.createDimension('ploc',num_order*num_loc_grid+1)
nc.createDimension('yloc',1)
w_in = nc.createVariable('w',np.dtype('float64').char,('ploc','yloc'))

w_in[:,:] = arrayw
nc.close 



### scatter plot
###

for smp_e in range(1) :
#smp_e=1
 nplt=200
###
 plt.figure()
 plt.scatter(fcst[ntime-nplt:ntime,smp_e], fcst_raw[ntime-nplt:ntime,smp_e]-anal[ntime-nplt:ntime,smp_e])
 plt.scatter(fcst[ntime-nplt:ntime,smp_e], fcst_raw[ntime-nplt:ntime,smp_e]-fcst[ntime-nplt:ntime,smp_e], c='red')
#plt.scatter(fcst[ntime-nplt:ntime,smp_e], fcst[ntime-nplt:ntime,smp_e]-anal[ntime-nplt:ntime,smp_e], c='green')
#plt.scatter(fcst[ntime-nplt:ntime,smp_e], fcst[ntime-nplt:ntime,smp_e]-anal[ntime-nplt:ntime,smp_e], c='red')
 plt.axhline(y=0, color='k', linestyle='--')
 plt.xlabel('forecast')
 plt.ylabel('forecast - analysis')
 plt.xlim()
 plt.ylim()
 figname='testw_'+str(smp_e)+'.png'
 plt.savefig(figname)
###

