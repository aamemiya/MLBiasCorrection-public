import os
import numpy as np
import model 
import param

### config
nx  = param.param_model['dimension'] 
nxx = param.param_model['dimension_coupled'] 
f   = param.param_model['forcing']
dt  = param.param_model['dt_coupled']
h   = param.param_model['h']
b   = param.param_model['b']
c   = param.param_model['c']
length = param.param_exp['spinup_length']
expdir = param.param_exp['expdir']

amp = 0

### spin up of coupled Lorenz96 system (nature run)

def spinup(inum):
  np.random.seed(inum)
  x0  = np.array(np.random.randn(nx),  dtype=np.float64)
  xx0 = np.array(np.random.randn(nxx), dtype=np.float64)
  l96 = model.Lorenz96_coupled(nx, nxx ,f ,dt, h, b, c, init_x = x0, init_xx = xx0, amp_const = amp)
  for i in range(length):
    l96.runge_kutta()
  l96.save_snap(expdir + '/spinup/init_coupled.nc')

os.system('mkdir -p ' +  expdir + '/spinup')
spinup(0)
