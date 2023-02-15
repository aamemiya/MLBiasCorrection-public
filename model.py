
import math                              # 数学ライブラリ
import numpy as np                       # 数値計算用の配列
import numpy.linalg as LA                # 行列計算ライブラリ
import netCDF4 


class Lorenz96:
  """
  This class provides the Lorenz96 model equations and the 4th-order Runge-Kutta solver.
  """
  def __init__(self, n = 40, f = 8, dt = 0.005, init_x = None , amp_const = 0 ):
    self.n = n
    self.f = f
    self.dt = dt
    self.amp = amp_const
    if init_x is not None:
      self.x = init_x
    else:
      self.x = np.zeros(n, dtype=np.float64)
    # temporary memory
    self.tmpx = np.zeros(n + 3, dtype=np.float64)
    self.tmpdx = np.zeros(n, dtype=np.float64)
    return
  
  def dx_dt(self, y):
    n = y.shape[0]
    m = n + 3
    self.tmpx[2:(m - 1)] = y[:]
    # cyclic boundary condition
    self.tmpx[0:2] = y[(n - 2):n]
    self.tmpx[m - 1] = y[0]
    #self.tmpdx = (self.tmpx[3:m] - self.tmpx[0:(m - 3)]) * self.tmpx[1:(m - 2)] - self.tmpx[2:(m - 1)] + self.f
    self.tmpdx[:] = self.tmpx[3:m]
    self.tmpdx   -= self.tmpx[0:(m - 3)]
    self.tmpdx   *= self.tmpx[1:(m - 2)]
    self.tmpdx   -= self.tmpx[2:(m - 1)]
    self.tmpdx   += self.f
    self.tmpdx   += self.amp * np.sin(2*np.pi*np.arange(n)/n)
    return self.tmpdx

  def runge_kutta(self):
    dx1 = self.dx_dt(self.x)
    x1  = self.x + dx1 * (self.dt * 0.5)
    dx2 = self.dx_dt(x1)
    x2  = self.x + dx2 * (self.dt * 0.5)
    dx3 = self.dx_dt(x2)
    x3  = self.x + dx3 * self.dt
    dx4 = self.dx_dt(x3)
    self.x += (dx1 + 2.0 * (dx2 + dx3) + dx4) * (self.dt / 6.0)
    return self.x

  def save_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'w',format='NETCDF3_CLASSIC')
    nc.createDimension('x',self.n)
    x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
    v_in = nc.createVariable('v',np.dtype('float64').char,('x'))
    x_in[:] = np.array(range(1,1+self.n))
    v_in[:] = self.x
    nc.close 

  def load_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'r',format='NETCDF4')
    self.x = np.array(nc.variables['v'][:], dtype=type(np.float64))
    nc.close 

class Lorenz96_add:
  """
  This class provides the Lorenz96 model equations and the 4th-order Runge-Kutta solver.
  """
  def __init__(self, n = 40, f = 8, dt = 0.005, init_x = None , amp_add = 0 , amp_add_2 = None , mode = None ):
    self.n = n
    self.f = f
    self.dt = dt
    self.amp = amp_add
    self.amp2 = f
    self.t = 0
    if amp_add_2 is not None:
      self.amp2 = amp_add_2
    self.mode = mode
    if init_x is not None:
      self.x = init_x
    else:
      self.x = np.zeros(n, dtype=np.float64)
    # temporary memory
    self.tmpx = np.zeros(n + 3, dtype=np.float64)
    self.tmpdx = np.zeros(n, dtype=np.float64)
    return
  
  def dx_dt(self, y):
    n = y.shape[0]
    m = n + 3
    self.tmpx[2:(m - 1)] = y[:]
    # cyclic boundary condition
    self.tmpx[0:2] = y[(n - 2):n]
    self.tmpx[m - 1] = y[0]
    #self.tmpdx = (self.tmpx[3:m] - self.tmpx[0:(m - 3)]) * self.tmpx[1:(m - 2)] - self.tmpx[2:(m - 1)] + self.f
    self.tmpdx[:] = self.tmpx[3:m]
    self.tmpdx   -= self.tmpx[0:(m - 3)]
    self.tmpdx   *= self.tmpx[1:(m - 2)]
    self.tmpdx   -= self.tmpx[2:(m - 1)]
    self.tmpdx   += self.f
    if self.mode == "const":
      self.tmpdx   += self.amp 
    if self.mode == "sin":
      self.tmpdx   += self.amp * np.sin(2*np.pi*np.arange(n)/n)
    if self.mode == "sint":
      self.tmpdx   += self.amp * np.sin(2*np.pi*self.t/self.amp2)
    if self.mode == "linear":
      self.tmpdx   += self.amp * self.tmpx[2:(m - 1)]
    if self.mode == "third_order":
      self.tmpdx   += self.amp * (self.tmpx[2:(m - 1)] - 1/(self.amp2)**2 * self.tmpx[2:(m - 1)]**3)
    if self.mode == "step":
      self.tmpdx   += self.amp * np.heaviside( self.tmpx[2:(m - 1)] - self.amp2, 0 )
    if self.mode == "advection":
      self.tmpdx   += self.amp * (self.tmpx[3:m] - self.tmpx[0:(m - 3)]) * self.tmpx[1:(m - 2)]
    return self.tmpdx

  def runge_kutta(self):
    dx1 = self.dx_dt(self.x)
    x1  = self.x + dx1 * (self.dt * 0.5)
    dx2 = self.dx_dt(x1)
    x2  = self.x + dx2 * (self.dt * 0.5)
    dx3 = self.dx_dt(x2)
    x3  = self.x + dx3 * self.dt
    dx4 = self.dx_dt(x3)
    self.x += (dx1 + 2.0 * (dx2 + dx3) + dx4) * (self.dt / 6.0)
    self.t += self.dt 
    return self.x

  def save_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'w',format='NETCDF3_CLASSIC')
    nc.createDimension('x',self.n)
    x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
    v_in = nc.createVariable('v',np.dtype('float64').char,('x'))
    x_in[:] = np.array(range(1,1+self.n))
    v_in[:] = self.x
    nc.close 

  def load_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'r',format='NETCDF4')
    self.x = np.array(nc.variables['v'][:], dtype=type(np.float64))
    nc.close 

class Lorenz96_coupled:
  """
  This class provides the coupled Lorenz96 model equations and the 4th-order Runge-Kutta solver.
  """
  def __init__(self, nx = 40, nxx = 1600, f = 8, dt = 0.005, h = 1, b = 10, c = 10, init_x = None, init_xx= None, amp_const = 0 ):
    self.nx  = nx
    self.nxx = nxx
    self.f   = f
    self.dt  = dt
    self.h   = h
    self.b   = b
    self.c   = c
    if init_x is not None:
      self.x = init_x
    else:
      self.x = np.zeros(nx, dtype=np.float64)
    if init_xx is not None:
      self.xx = init_xx
    else:
      self.xx = np.zeros(nxx, dtype=np.float64)
    # temporary memory
    self.tmpx = np.zeros(nx + 3, dtype=np.float64)
    self.tmpdx = np.zeros(nx, dtype=np.float64)
    self.tmpxx = np.zeros(nxx + 3, dtype=np.float64)
    self.tmpdxx = np.zeros(nxx, dtype=np.float64)
    # coupling function
    self.coupler = np.zeros((self.nx, self.nxx), dtype=np.float64)
    nratio = nxx / nx
    for i in range(nx):
      self.coupler[i,int(nratio*i):int(nratio*(i+1))] = 1.0
    return
  
  def dx_dt(self, y, yy):
    nx = y.shape[0]
    nxx = yy.shape[0]

    m = nx + 3
    self.tmpx[2:(m - 1)] = y
    # cyclic boundary condition
    self.tmpx[0:2] = y[(nx - 2):nx]
    self.tmpx[m - 1] = y[0]

    mm = nxx + 3
    self.tmpxx[1:(mm - 2)] = yy
    # cyclic boundary condition
    self.tmpxx[0] = yy[nxx - 1]
    self.tmpxx[(mm - 2):mm] = yy[0:2]

    #self.tmpdx = (self.tmpx[3:m] - self.tmpx[0:(m - 3)]) * self.tmpx[1:(m - 2)] - self.tmpx[2:(m - 1)] + self.f
    self.tmpdx[:] = self.tmpx[3:m]
    self.tmpdx   -= self.tmpx[0:(m - 3)]
    self.tmpdx   *= self.tmpx[1:(m - 2)]
    self.tmpdx   -= self.tmpx[2:(m - 1)]
    self.tmpdx   += self.f
    self.tmpdx   -= self.h * self.c / self.b * np.dot(self.coupler,yy.astype('float64'))

    self.tmpdxx[:] = self.tmpxx[0:(mm - 3)]
    self.tmpdxx   -= self.tmpxx[3:mm]
    self.tmpdxx   *= self.c * self.b * self.tmpxx[2:(mm - 1)]
    self.tmpdxx   -= self.c * self.tmpxx[1:(mm - 2)]
    self.tmpdxx   += self.h * self.c / self.b * np.dot(y.astype('float64'),self.coupler)

    return self.tmpdx, self.tmpdxx

  def runge_kutta(self):
#    xxx = self.x, self.xx
#    dx1 = self.dx_dt(xxx)
#    x1  = xxx + dx1 * (self.dt * 0.5)
#    dx2 = self.dx_dt(x1)
#    x2  = xxx + dx2 * (self.dt * 0.5)
#    dx3 = self.dx_dt(x2)
#    x3  = xxx + dx3 * self.dt
#    dx4 = self.dx_dt(x3)
#    xxx += (dx1 + 2.0 * (dx2 + dx3) + dx4) * (self.dt / 6.0)
#    self.x , self.xx = xxx
    dx1,dxx1 = self.dx_dt(self.x,self.xx)
    x1  = self.x + dx1 * (self.dt * 0.5)
    xx1 = self.xx + dxx1 * (self.dt * 0.5)
    dx2,dxx2 = self.dx_dt(x1,xx1)
    x2  = self.x + dx2 * (self.dt * 0.5)
    xx2 = self.xx + dxx2 * (self.dt * 0.5)
    dx3,dxx3 = self.dx_dt(x2,xx2)
    x3  = self.x + dx3 * self.dt
    xx3  = self.xx + dxx3 * self.dt
    dx4,dxx4 = self.dx_dt(x3,xx3)
#    print('runge_kutta bef', self.x[0:2], self.xx[0:2])
    self.x += (dx1 + 2.0 * (dx2 + dx3) + dx4) * (self.dt / 6.0)
    self.xx += (dxx1 + 2.0 * (dxx2 + dxx3) + dxx4) * (self.dt / 6.0)
#    print('runge_kutta aft', self.x[0:2], self.xx[0:2])

    return self.x, self.xx

  def save_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'w',format='NETCDF3_CLASSIC')
    nc.createDimension('x',self.nx)
    nc.createDimension('xx',self.nxx)
    x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
    xx_in = nc.createVariable('xx',np.dtype('float64').char,('xx'))
    v_in = nc.createVariable('v',np.dtype('float64').char,('x'))
    vv_in = nc.createVariable('vv',np.dtype('float64').char,('xx'))
    x_in[:] = np.array(range(1,1+self.nx))
    xx_in[:] = np.array(range(1,1+self.nxx)) * self.nx / self.nxx ### TORI AEZU 
    v_in[:] = self.x
    vv_in[:] = self.xx
    nc.close 

  def load_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'r',format='NETCDF3_CLASSIC')
    self.x  = np.array(nc.variables['v'][:], dtype=type(np.float64))
    self.xx = np.array(nc.variables['vv'][:], dtype=type(np.float64))
    nc.close 


