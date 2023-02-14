#------------------------------------------------
import os
import sys
import math
import numpy as np
import numpy.linalg as LA
import netCDF4
import model
import letkf_fullmodel as letkf
import param
import bias_correction as BC

#------------------------------------------------
### config
nx  = param.param_model['dimension'] 
f   = param.param_model['forcing']
dt  = param.param_model['dt_coupled']

nxx  = param.param_model['dimension_coupled'] 
h   = param.param_model['h']
b   = param.param_model['b']
c   = param.param_model['c']

length = param.param_exp['exp_length']
expdir = param.param_exp['expdir']
obsdir = param.param_exp['obs_type']
dadir  = param.param_exp['da_type']

nmem        = param.param_exp['ensembles']
dt_nature   = param.param_exp['dt_nature']
dt_assim    = param.param_exp['dt_assim']

loc_scale  = param.param_letkf['localization_length_scale']
loc_cutoff = param.param_letkf['localization_length_cutoff']
fact_infl  = param.param_letkf['inflation_factor'] 

bc_type = param.param_bc['bc_type']
bc_alpha = param.param_bc['alpha']
bc_gamma = param.param_bc['gamma']

intv_nature=int(dt_nature/dt)
intv_assim=int(dt_assim/dt)

dadir='fcst_from_nature'
tag='fullmodel'
amp_perturb=0.1

length=80
nsmpmax=100
intv=250
inittime=4501   ### spinup
#------------------------------------------------

letkf = letkf.LETKF(model.Lorenz96_coupled, nx, nxx, f, dt = dt, h= h, b = b, c= c, k = nmem, localization_len = loc_scale, localization_cut = loc_cutoff , inflation = fact_infl)

nc = netCDF4.Dataset(expdir + '/nature_full.nc','r',format='NETCDF4')
nature = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
nature_vv = np.array(nc.variables['vv'][:], dtype=type(np.float64)).astype(np.float32)
time_nature = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

# initial ensemble perturbation
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/nocorr/assim.nc','r',format='NETCDF4')
analysis = np.array(nc.variables['va'][:], dtype=type(np.float64)).astype(np.float32)
time_assim = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close

assim_length=len(time_assim)

ismp=0
rmsesmp=[]
sprdsmp=[]
while ((inittime+length <= assim_length) and (ismp < nsmpmax)) :
  
  print("inittime ",inittime)
  
  for i in range(nmem):
#    letkf.ensemble[i].x = analysis[inittime,i,:]
    letkf.ensemble[i].x = nature[inittime,:] + np.random.randn(nx)*amp_perturb
    letkf.ensemble[i].xx = np.copy(nature_vv[inittime,:])

  inittime_nature = int(np.where(time_nature == time_assim[inittime])[0])
  ntime_nature=len(time_nature)

  bc=BC.BiasCorrection(bc_type,nx,bc_alpha,bc_gamma)

 ### spinup of LSTM
  if bc_type == 'tf':
    for i in range(100):  
      xftemp = bc.correct(analysis[inittime-99+i,:,:])
 
  xf=[]
  xfm=[]
  xfm_raw=[]
  rmse=[]
  sprd=[]

# init value
  xf.append(letkf.members())
  xfm.append(letkf.mean())
  xfm_raw.append(letkf.mean())
  rmse.append(math.sqrt(((letkf.mean()-nature[inittime_nature])**2).sum() /nx))
  sprd.append(math.sqrt((letkf.sprd()**2).sum()/nx))
# 

  length_nature=int(length*dt_assim/dt_nature)

# MAIN LOOP
  for step in range(inittime_nature, inittime_nature+length_nature):
    for i in range(intv_nature):
      letkf.forward()
      if param.param_bc['correct_step'] is not None:
        for j in range(nmem):
          fact=1.0/intv_assim
          letkf.ensemble[j].x = fact * bc.correct(letkf.ensemble[j].x) + (1.0-fact) * letkf.ensemble[j].x
   
    if (np.count_nonzero(time_assim == time_nature[step+1])): 
      step_assim=int(np.where(time_assim == time_nature[step+1])[0])
      xfmtemp=letkf.mean()
      xfmtempb=letkf.mean()
      xfm_raw.append(xfmtemp)
      if param.param_bc['correct_step'] is not None:
         for j in range(nmem):
            fact=1.0/intv_assim
            letkf.ensemble[j].x = fact * bc.correct(letkf.ensemble[j].x) + (1.0-fact) * letkf.ensemble[j].x
      else:
#       for i in range(nmem):
#           letkf.ensemble[i].x = bc.correct(letkf.ensemble[i].x)
         net_array = np.stack([letkf.ensemble[j].x for j in range(nmem)])
#       xftemp = bc.correct(net_array).reshape(nmem, nx)
         xftemp = bc.correct(net_array)
         for j in range(nmem):
           letkf.ensemble[j].x = xftemp[j]
 
 
      xf.append(letkf.members())
      xfmtemp=letkf.mean()
      xfm.append(xfmtemp)

      rmsetemp = math.sqrt(((letkf.mean()-nature[step+1])**2).sum() /nx)
      rmseraw = math.sqrt(((xfmtempb-nature[step+1])**2).sum() /nx)
      sprdtemp = math.sqrt((letkf.sprd()**2).sum()/nx)
#    if ( round(step/1,4).is_integer() and step-inittime_nature < 10 ):
      if ( step-inittime_nature < 10 ):
        print('time ', round(time_nature[step],4),' RMSE ', round(rmsetemp,4), ' RMSE(raw) ', round(rmseraw,4), ' SPRD ', round(sprdtemp,4) )
#          print('time ', round(time_obs[step_obs],4),' RMSE ', rmse,' SPRD ',sprd)
      rmse.append(math.sqrt(((letkf.mean()-nature[step+1])**2).sum() /nx))
      sprd.append(math.sqrt((letkf.sprd()**2).sum()/nx))
#  if ( round(icount/10,4).is_integer() ):
#  print("time {0:10.4f}, RMSE , {1:8.4f}, SPRD ,{2:8.4f}".format(round(time_nature[step],4),rmse[step-inittime_nature],sprd[step-inittime_nature]))
        
#print("done")

  rmsesmp.append(rmse)
  sprdsmp.append(sprd)

  ismp+=1 
  inittime+=intv

rmsesmp  = np.array(rmsesmp,  dtype=np.float64)
sprdsmp  = np.array(sprdsmp,  dtype=np.float64)


print(rmsesmp.shape)

if (os.path.isfile(expdir + '/' + obsdir + '/' + dadir + '/stats_' + tag + '.nc')) :
    print('overwrite')
    nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/stats_' + tag + '.nc','r+',format='NETCDF4')
    nsmp=len(nc.variables['rmse'])
    nc.variables['rmse'][nsmp:,:]=rmsesmp
    nc.variables['sprd'][nsmp:,:]=sprdsmp
    nc.close
else : 
    print('create')
    nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/stats_' + tag + '.nc','w',format='NETCDF4')
    nc.createDimension('t',length+1)
    nc.createDimension('s',None)
    rmse_in = nc.createVariable('rmse',np.dtype('float64').char,('s','t'))
    sprd_in = nc.createVariable('sprd',np.dtype('float64').char,('s','t'))
    t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
    rmse_in[:,:] = rmsesmp
    sprd_in[:,:] = sprdsmp
    t_in[:] = np.round(time_nature[:length+1],4)
    nc.close


