import os
import netCDF4 
import numpy as np
import model 
import param

### config
expdir = param.param_exp['expdir']
obsdir = param.param_exp['obs_type']
dt_obs = param.param_exp['dt_obs']

obs_err_std = param.param_letkf['obs_error']

obsnum = param.param_letkf['obs_number']
miss = param.param_letkf['missing_value']

### random observation masking 

def obsmask(data) :
    if (obsnum != data.shape[0]):
      rarray=np.random.rand(data.shape[0])
      for i in range(data.shape[0]-obsnum): 
        j=rarray.argmax()
        data[j]=miss
        rarray[j]=0
    return data

### load nature run and create obs
nc = netCDF4.Dataset(expdir + '/nature.nc','r',format='NETCDF3_CLASSIC')
nature = np.array(nc.variables['v'][:], dtype=type(np.float64))
time_nature = np.array(nc.variables['t'][:], dtype=type(np.float64))
nc.close 

### *** observation operator is not considered yet ***
list_obs = nature + np.random.randn(nature.shape[0], nature.shape[1]) * obs_err_std
n=nature.shape[1]

obs=[]
time_obs=[]

ntime_nature=len(time_nature)
for step in range(0,ntime_nature):
  time_now = time_nature[step]

  if (round(time_now/dt_obs,4).is_integer()):
    time_obs.append(round(time_now,6))
    obs.append(obsmask(list_obs[step]))

os.system('mkdir -p ' + expdir + '/' + obsdir)
 
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/obs.nc','w',format='NETCDF3_CLASSIC')
nc.createDimension('y',n)
nc.createDimension('t',None)
y_in = nc.createVariable('y',np.dtype('float64').char,('y'))
t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
vy_in = nc.createVariable('vy',np.dtype('float64').char,('t','y'))
y_in[:] = np.array(range(1,1+n))
t_in[:] = time_obs
vy_in[:,:] = obs
nc.close 

