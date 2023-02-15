#------------------------------------------------
import os
import numpy as np
import netCDF4
#import param
import matplotlib.pyplot as plt

#------------------------------------------------
### config
nx=8

expdir='../DATA_test/coupled_A13/0001'

obsdir='obs_p8_010'
obsdir2='obs_p6_010'

dadir=['nocorr','linear','linear4','Dense_single','Dense','SimpleRNN','GRU','LSTM']

legends = ['None','Linear','Linear4th','Dense_single','Dense','SimpleRNN','GRU','LSTM']

lstyles = ['solid','dotted','dotted','dashed','dashed','dashdot','dashdot','dashdot']

colors = ['tab:red','tab:orange','tab:brown','tab:purple','tab:pink','tab:green','tab:cyan','tab:blue']


#------------------------------------------------

nexp=len(dadir)

rmse=[]
sprd=[]
rmse2=[]
sprd2=[]


values=[]
errors=[]
plows=[]
phighs=[]
values2=[]
errors2=[]
plows2=[]
phighs2=[]

vth=1.0
vkss=[]
vkss2=[]
# load nature and observation data
for i in range(nexp):
  nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + dadir[i] + '/stats.nc','r',format='NETCDF4')
  rmse.append(np.array(nc.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32))
  sprd.append(np.array(nc.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32))
  nc2 = netCDF4.Dataset(expdir + '/' + obsdir2 +'/' + dadir[i] + '/stats.nc','r',format='NETCDF4')
  rmse2.append(np.array(nc2.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32))
  sprd2.append(np.array(nc2.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32))
  if (i == 0): 
    time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
    dt=time[1]-time[0]
  nc.close 

  vks=[]
  vks2=[]

  for j in range(rmse[i].shape[0]):
    for k in range(rmse[i][j].shape[0]-1):
      if (rmse[i][j][k] < vth) and (rmse[i][j][k+1] > vth) :
        vks.append(dt*(float(k) + (vth-rmse[i][j][k])/(rmse[i][j][k+1]-rmse[i][j][k])))
        break

  for j in range(rmse2[i].shape[0]):
    for k in range(rmse2[i][j].shape[0]-1):
      if (rmse2[i][j][k] < vth) and (rmse2[i][j][k+1] > vth) :
        vks2.append(dt*(float(k) + (vth-rmse2[i][j][k])/(rmse2[i][j][k+1]-rmse2[i][j][k])))
        break

  vkss.append(np.array(vks))
  vkss2.append(np.array(vks2))

vkss=np.array(vkss)
vkss2=np.array(vkss2)

for i in range(1,nexp): 
  vkss[i] = vkss[i]/vkss[0]
  vkss2[i] = vkss2[i]/vkss2[0]
  values.append(np.mean(vkss[i]))
  errors.append(np.std(vkss[i]))
  plows.append(np.percentile(vkss[i],10.))
  phighs.append(np.percentile(vkss[i],90.))
  values2.append(np.mean(vkss2[i]))
  errors2.append(np.std(vkss2[i]))
  plows2.append(np.percentile(vkss2[i],10.))
  phighs2.append(np.percentile(vkss2[i],90.))
  print(legends[i],np.mean(vkss[i]),np.std(vkss[i]),np.mean(vkss2[i]),np.std(vkss2[i]),np.percentile(vkss2[i],10.),np.percentile(vkss2[i],90.))
#quit()

ntime=len(time)

#doubling_time=0.2*2.1 ### 2.1 day

#refy=np.array((1,max(rmse_plot[0])))
#refx=np.array((0,doubling_time*np.log2(refy[1])))


fig, ax = plt.subplots()

pcts=[]
pcts.append(np.array(values)-np.array(plows))
pcts.append(np.array(phighs)-np.array(values))
pcts=np.array(pcts)
pcts2=[]
pcts2.append(np.array(values)-np.array(plows2))
pcts2.append(np.array(phighs2)-np.array(values))
pcts2=np.array(pcts2)

#ax.add_patch(Rectangle(xy=(2.5,0.2),width=2.0,height=1.0 , facecolor='gainsboro', edgecolor=None ))
plt.axvline(x=4.5, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=2.5, color='gray', linestyle='--', linewidth=1)

#ax.bar(np.arange(1.0,float(nexp)), np.array(values[:]), width=0.5, color=colors[1:])
#ax.errorbar(np.arange(1.0,float(nexp)), np.array(values[:]), fmt="x",yerr = np.array(errors[:]), capsize=2.0, ecolor='black', markeredgecolor = "red", linestyle="")
#ax.errorbar(np.arange(1.1,float(nexp)+0.1), np.array(values2[:]), fmt="x",yerr = np.array(errors2[:]), capsize=2.0, ecolor='grey', markeredgecolor = "red", linestyle="")
ax.errorbar(np.arange(1.0,float(nexp)), np.array(values[:]), fmt="x",yerr = pcts, capsize=2.0, ecolor='black', markeredgecolor = "red", linestyle="")
ax.errorbar(np.arange(1.1,float(nexp)+0.1), np.array(values2[:]), fmt="x",yerr = pcts2, capsize=2.0, ecolor='grey', markeredgecolor = "red", linestyle="")

plt.axhline(y=1, color='k', linestyle='--')

ax.set_ylabel('Ratio of lead time',fontsize=12)

ax.set_ylim(0.5,2.5)
ax.set_xlim(0.5,float(nexp)-0.5)

ax.set_xticklabels(legends[:], rotation= 30, ha="right")
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.25)

fig.savefig('png/fig10.png')
#

