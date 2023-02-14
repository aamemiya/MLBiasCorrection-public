#------------------------------------------------
import os
import math
import numpy as np
import numpy.linalg as LA
import netCDF4
import model
import letkf
import param
import bias_correction as BC

#------------------------------------------------
### config
nx  = param.param_model['dimension'] 
f   = param.param_model['forcing']
dt  = param.param_model['dt']

length = param.param_exp['exp_length']
expdir = param.param_exp['expdir']
obsdir = param.param_exp['obs_type']
dadir  = param.param_exp['da_type']

obs_err_std = param.param_letkf['obs_error']
nmem        = param.param_exp['ensembles']
dt_nature   = param.param_exp['dt_nature']
dt_obs      = param.param_exp['dt_obs']
dt_assim    = param.param_exp['dt_assim']

loc_scale  = param.param_letkf['localization_length_scale']
loc_cutoff = param.param_letkf['localization_length_cutoff']
fact_infl  = param.param_letkf['inflation_factor'] 
miss = param.param_letkf['missing_value']
fact_infl_add  = param.param_letkf['additive_inflation_factor'] 

bc_type = param.param_bc['bc_type']
bc_alpha = param.param_bc['alpha']
bc_gamma = param.param_bc['gamma']

intv_nature=int(dt_nature/dt)
intv_assim=int(dt_assim/dt)

#------------------------------------------------

def qc(h,data,r) :
    h_qc=[]
    data_qc=[]
    r_qc=[]
    for i in range(data.shape[0]) : 
      if (data[i] != miss) : 
        h_qc.append(h[i])
        data_qc.append(data[i])
        r_qc.append(r[i])
    h_qc=np.array(h_qc,  dtype=np.float64)
    data_qc=np.array(data_qc,  dtype=np.float64)
    r_qc=np.array(r_qc,  dtype=np.float64)
 
    return h_qc,data_qc,r_qc

#------------------------------------------------


letkf = letkf.LETKF(model.Lorenz96, nx, f, k = nmem, localization_len = loc_scale, localization_cut = loc_cutoff , inflation = fact_infl, add_inflation = fact_infl_add)
# initial ensemble perturbation
for i in range(nmem):
  nc = netCDF4.Dataset(expdir + '/spinup/init'+ '{0:02d}'.format(i) + '.nc','r',format='NETCDF4')
  x0 = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
  nc.close 
  letkf.ensemble[i].x = x0.copy()

# load nature and observation data
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/obs.nc','r',format='NETCDF4')
obs = np.array(nc.variables['vy'][:], dtype=type(np.float64)).astype(np.float32)
time_obs = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

nc = netCDF4.Dataset(expdir + '/nature.nc','r',format='NETCDF4')
nature = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
time_nature = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

# set observation error covariance matrix (diagonal elements only)
r = np.ones(nx, dtype=np.float64) * obs_err_std
# set observation operator matrix (use identity)
h = np.identity(nx, dtype=np.float64)

xfm = []
xfm_raw = []
xam = []
xf  = []
xa  = []
time_assim = []
ntime_nature=len(time_nature)

bc=BC.BiasCorrection(bc_type,nx,bc_alpha,bc_gamma)

icount=-100
rmse_mean=0

plot_count=0

# MAIN LOOP
for step in range(min(ntime_nature,length-1)):
  for i in range(intv_nature):
    letkf.forward()
    if param.param_bc['correct_step'] is not None:
      for i in range(nmem):
        fact=1.0/intv_assim
        letkf.ensemble[i].x = fact * bc.correct(letkf.ensemble[i].x) + (1.0-fact) * letkf.ensemble[i].x
  if (np.count_nonzero(time_obs == time_nature[step])): 
    step_obs=int(np.where(time_obs == time_nature[step])[0])
    if (round(time_obs[step_obs]/dt_assim,2).is_integer()):  
      if (step + 1) < length:
        xfmtempb=letkf.mean()
        xfm_raw.append(xfmtempb)
        if (step > 0): 
          if param.param_bc['correct_step'] is not None:
            for i in range(nmem):
              fact=1.0/intv_assim
              letkf.ensemble[i].x = fact * bc.correct(letkf.ensemble[i].x) + (1.0-fact) * letkf.ensemble[i].x
          else:
             net_array = np.stack([letkf.ensemble[i].x for i in range(nmem)])
             xftemp = bc.correct(net_array)

             for i in range(nmem):
               letkf.ensemble[i].x = xftemp[i]
 
        xf.append(letkf.members())
        xfmtemp=letkf.mean()
        xfm.append(xfmtemp)
     
        xa.append(letkf.analysis(*qc(h, obs[step_obs], r)))
        xamtemp=letkf.mean()
        xam.append(xamtemp)
        time_assim.append(round(time_obs[step_obs],4))

        if param.param_bc['offline'] is None  : bc.train(xfmtemp,xamtemp)  

        dfa  = np.sqrt(np.mean(np.power(xfmtemp-xamtemp, 2)))
        dfar = np.sqrt(np.mean(np.power(xfmtempb-xamtemp, 2)))
        dfn  = np.sqrt(np.mean(np.power(xfmtempb-nature[step], 2)))

        rmse = math.sqrt(((letkf.mean()-nature[step])**2).sum() /nx)
        sprd = math.sqrt((letkf.sprd()**2).sum()/nx)
        if ( round(icount/9,4).is_integer() ):
          print('time ', round(time_obs[step_obs],4),' RMSE ', round(rmse,4), ' f-a ', round(dfa,4), ' f-a(raw) ', round(dfar,4), ' f-n(raw) ', round(dfn,4), ' SPRD ', round(sprd,4) )
#          print('time ', round(time_obs[step_obs],4),' RMSE ', rmse,' SPRD ',sprd)
        icount = icount+1

        if (icount > 0):
          rmse_mean = (rmse + (icount-1) * rmse_mean)  / icount
#          if ( round((icount-1)/1,4).is_integer() ):
#            print(' RMSE ', rmse ,' RMSE_mean ', rmse_mean )

time_assim = np.array(time_assim, dtype=np.float64)
xf  = np.array(xf,  dtype=np.float64)
xa  = np.array(xa,  dtype=np.float64)
xfm_raw = np.array(xfm_raw, dtype=np.float64)
xfm = np.array(xfm, dtype=np.float64)
xam = np.array(xam, dtype=np.float64)

print('rmse mean =' , rmse_mean)

print("done")

#rmse = []
#for xx in x_letkf10:
#  if xx[0] < nature.shape[0]:
#    rmse.append((xx[0], math.sqrt(((nature[xx[0]] - xx[1]) ** 2).sum() / n)))
#rmse = np.array(rmse) 

os.system('mkdir -p ' + expdir + '/' + obsdir + '/' + dadir)
 
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/assim.nc','w',format='NETCDF4')
nc.createDimension('x',nx)
nc.createDimension('e',nmem)
nc.createDimension('t',None)
x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
e_in = nc.createVariable('e',np.dtype('float64').char,('e'))
t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
va_in = nc.createVariable('va',np.dtype('float64').char,('t','e','x'))
vf_in = nc.createVariable('vf',np.dtype('float64').char,('t','e','x'))
vam_in = nc.createVariable('vam',np.dtype('float64').char,('t','x'))
vfm_in = nc.createVariable('vfm',np.dtype('float64').char,('t','x'))
vfm_raw_in = nc.createVariable('vfm_raw',np.dtype('float64').char,('t','x'))
x_in[:] = np.array(range(1,1+nx))
e_in[:] = np.array(range(1,1+nmem))
t_in[:] = np.round(time_assim,4)
va_in[:,:] = xa
vf_in[:,:] = xf
vam_in[:,:] = xam
vfm_in[:,:] = xfm
vfm_raw_in[:,:] = xfm_raw
nc.close 
