
import math                               
import numpy as np                    
import numpy.linalg as LA             

class LETKF:
  def __init__(self, model, n = 40, f = 8, dt = 0.005, amp_const = 0, k = 20, localization_cut = 5, localization_len = 3, inflation = 1.0, add_inflation = 0):
    """
    model: model constructor
    n: model size
    f: external forcing
    dt: time interval of model time integration
    amp_const: amplitude of constant inhomogeneous additional term  
    n, f, dt, amp_const will be passed to the model constructor
    k: ensemble size
    localization_cut: localization patch cutoff radius
    localization_len: localization patch length scale (exp(-0.5*(r/len)**2))
    inflation: multiplicative covariance inflation
    add_inflation: additive covariance inflation
    """
    self.n = n
    self.f = f
    self.dt = dt
    self.k = k
    self.amp = amp_const
    self.localization_len = localization_len
    self.localization_cut = localization_cut
    self.infl = math.sqrt(inflation)
   
    self.add_infl = add_inflation

    # models
    self.ensemble = []
    for i in range(k):
      self.ensemble.append(model(n, f, dt, amp_const=amp_const))
    return
  
  def forward(self):
    """
    This function updates the model state.
    """
    for i in range(self.k):
      self.ensemble[i].runge_kutta()
    return

  def members(self):
    x_members=np.zeros((self.k, self.n), dtype=np.float64)
    for i in range(0, self.k):
      x_members[i,:] = self.ensemble[i].x.copy()
    return x_members

  def mean(self):
    x_mean = self.ensemble[0].x.copy()
    for i in range(1, self.k):
      x_mean += self.ensemble[i].x
    x_mean /= self.k
    return x_mean

  def sprd(self):
    x_mean = self.mean()
    x_sprd = (self.ensemble[0].x.copy() - x_mean)**2
    for i in range(1, self.k):
      x_sprd += (self.ensemble[i].x.copy() - x_mean)**2
    x_sprd /= self.k
    for j in range(self.n):
      x_sprd[j] = math.sqrt(x_sprd[j])
    return x_sprd

  def analysis(self, h, y, r):
    """
    This function performs LETKF.
    h: observation operator matrix
    y: a vector of observations
    r: observation error covariance matrix
    """

    # forecast ensemble mean
    x = self.mean()[:, np.newaxis]
    # forecast ensemble perturbation divided by sqrt(self.k - 1)
    z = np.zeros((self.n, self.k), dtype=np.float64)
    for i in range(self.k):
      z[:, i] = self.ensemble[i].x
    
    z -= x
    z /= math.sqrt(self.k - 1)
    # forecast ensemble perturbation in the observation space
    hzz = h @ z

    hx0 = h @ x
    hx = np.zeros((y.shape[0], self.k), dtype=np.float64)
    for i in range(self.k):
      hx[:, i] = hx0[:, 0]
    # mismatch between the forecast ensemble mean and observation (called "innovation")
    y_hx = y[:, np.newaxis] - hx

    # analysis at each location
    for ia in range(self.n):
      x_ia = x[ia, :] # ensemble mean at this location
      z_ia = z[ia, :] # ensemble perturbation at this location

      # search local observations within the localization radius
      obs_loc, factor_loc = self.search_obs(ia, h)

      n_obs_loc = len(obs_loc)
      if n_obs_loc == 0:
        continue
      hz_loc = hzz[obs_loc] # forecast ensemble perturbation in the observation space at this location
      y_hx_loc = y_hx[obs_loc, :] # innovation for local observations
      # inverse of observation error covariance for local observations
      if r.ndim == 1: # r is a 1-D array (only the diagonal elements are provided)
        r_inv_loc = np.zeros((n_obs_loc, n_obs_loc), dtype=np.float64)
        for i in range(n_obs_loc):
          r_inv_loc[i, i] = 1.0 / r[obs_loc[i]] * factor_loc[i]
      else:
        r_inv_loc = LA.inv(r[obs_loc, :][:, obs_loc])

      # only the value at this location is updated
      x_a = self.letkf_core(x_ia, z_ia, hz_loc, y_hx_loc, r_inv_loc)
      for i in range(self.k):
        self.ensemble[i].x[ia] = x_a[i].real
    return self.members()

  def search_obs(self, ia, h):
    patch = []
    factor_x = []
    for i in range(ia - self.localization_cut, ia + self.localization_cut + 1):
      patch.append(i % self.n)
      factor_x.append(np.exp( -0.5 * ((i - ia) / self.localization_len)**2 ))
    obs_loc = []
    factor_loc = []
    for i in range(h.shape[0]):
      if np.max(np.abs(h[i, patch])) > 0:
        obs_loc.append(i)
        factor_loc.append( h[i,patch] @ factor_x )
    return obs_loc, factor_loc

  def letkf_core(self, x, z, hz, y_hx, r_inv):
    """
    x: forecast ensemble mean
    z: forecast ensemble perturbation divided by sqrt(self.k - 1)
    hz: z in the observation space
    y_hz: innovation
    r_inv: inverse of observation error covariance
    """
    # ensemble transform
    hzT_r_inv = hz.T @ r_inv
    u_d_u = hzT_r_inv @ hz
    for i in range(u_d_u.shape[0]):
      u_d_u[i, i] += 1.0
    d, u = LA.eig(u_d_u)
    d_inv = np.identity(d.shape[0], dtype=np.float64)
    d_inv_sqrt = np.identity(d.shape[0], dtype=np.float64)
    for i in range(d.shape[0]):
      d_inv[i, i] = 1.0 / d[i].real
      d_inv_sqrt[i, i] = 1.0 / math.sqrt(d[i].real)

    w = u @ d_inv @ u.T @ hzT_r_inv @ y_hx  # mean update
    w += (u @ d_inv_sqrt @ u.T) * math.sqrt(self.k - 1) # ensemble update

    if self.add_infl > 0:
       z_add = np.random.randn(self.k)
       z_add = z_add - np.mean(z_add)
       x_a = (z @ w) * self.infl + z_add * self.add_infl + x # additive/covariance inflation is included here
    else : 
       x_a = (z @ w) * self.infl + x # covariance inflation is included here
    return x_a

