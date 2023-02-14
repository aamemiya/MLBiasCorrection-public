#------------------------------------------------
import os
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#------------------------------------------------
### config
nx=8

expdir='../DATA_test/coupled_A13/0001'

obsdir='obs_p8_010'
obsdir2='obs_p6_010'
#
tfile='./data/loss_p8.txt'

tlen=29899
tsta=100
tend=tsta+tlen

#------------------------------------------------

titles=['nocorr','linear','linear4','Dense_single','Dense','SimpleRNN','GRU','LSTM']
dadir=['nocorr','linear','linear4','Dense_single','Dense','SimpleRNN','GRU','LSTM']


values=[]
errors=[]
values2=[]
errors2=[]
colors=[]
vdifs=[]
vdifs2=[]
plows=[]
phighs=[]
plows2=[]
phighs2=[]

nexp=len(titles)

ncn = netCDF4.Dataset(expdir + '/nature.nc','r',format='NETCDF4')
nature=np.array(ncn.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
for i in range(nexp):
  nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + dadir[i] + '/assim.nc','r',format='NETCDF4')
  nc2 = netCDF4.Dataset(expdir + '/' + obsdir2 +'/' + dadir[i] + '/assim.nc','r',format='NETCDF4')
  vam=np.array(nc.variables['vam'][:], dtype=type(np.float64)).astype(np.float32)
  vam2=np.array(nc2.variables['vam'][:], dtype=type(np.float64)).astype(np.float32)
  vdif=vam[tsta:tend]-nature[tsta:tend]
  vdif=np.sqrt(np.mean(vdif**2,axis=-1))
  vdif2=vam2[tsta:tend]-nature[tsta:tend]
  vdif2=np.sqrt(np.mean(vdif2**2,axis=-1))


  vdifs.append(vdif)
  vdifs2.append(vdif2)

  if (i == 0): 
    time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
  nc.close 

  if np.mod(i,2) == 0 : 
    colors.append('grey')
  else:
    colors.append('lightgrey')

ncn.close

vdifs=np.array(vdifs)
vdifs2=np.array(vdifs2)

for i in range(1,nexp): 
  vdifs[i] = vdifs[i]/vdifs[0]
  values.append(np.mean(vdifs[i]))
  errors.append(np.std(vdifs[i]))
  vdifs2[i] = vdifs2[i]/vdifs2[0]
  values2.append(np.mean(vdifs2[i]))
  errors2.append(np.std(vdifs2[i]))
  plows.append(np.percentile(vdifs[i],10.))
  phighs.append(np.percentile(vdifs[i],90.))
  plows2.append(np.percentile(vdifs2[i],10.))
  phighs2.append(np.percentile(vdifs2[i],90.))



#  print(legends[i],np.mean(vdifs[i]),np.std(vdifs[i]))
#quit()


values=np.array(values)
errors=np.array(errors)
values2=np.array(values2)
errors2=np.array(errors2)
print(np.arange(1.0,float(nexp)))
print(np.array(values))
print(np.array(errors))
print(np.array(titles))
print(np.array(colors))
pcts=[]
pcts.append(np.array(values)-np.array(plows))
pcts.append(np.array(phighs)-np.array(values))
pcts=np.array(pcts)
pcts2=[]
pcts2.append(np.array(values)-np.array(plows2))
pcts2.append(np.array(phighs2)-np.array(values))
pcts2=np.array(pcts2)


fig, ax = plt.subplots()

#ax.add_patch(Rectangle(xy=(2.5,0.2),width=2.0,height=1.0 , facecolor='gainsboro', edgecolor=None ))
plt.axvline(x=4.5, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=2.5, color='gray', linestyle='--', linewidth=1)

#ax.bar(np.arange(1.0,float(nexp)), np.array(values[:]), width=0.5, color=colors[1:])
#ax.errorbar(np.arange(1.0,float(nexp)), np.array(values[:]), fmt="x",yerr = np.array(errors[:]), capsize=2.0, ecolor='black', markeredgecolor = "red", linestyle="")
#ax.errorbar(np.arange(1.1,float(nexp)+0.1), np.array(values2[:]), fmt="x",yerr = np.array(errors2[:]), capsize=2.0, ecolor='grey', markeredgecolor = "red", linestyle="")
ax.errorbar(np.arange(1.0,float(nexp)), np.array(values[:]), fmt="x",yerr = pcts, capsize=2.0, ecolor='black', markeredgecolor = "red", linestyle="")
ax.errorbar(np.arange(1.1,float(nexp)+0.1), np.array(values2[:]), fmt="x",yerr = pcts2, capsize=2.0, ecolor='dimgrey', markeredgecolor = "red", linestyle="")


plt.axhline(y=1, color='k', linestyle='--')

ax.set_ylabel('Relative analysis error',fontsize=12)

ax.set_ylim(0.2,1.2)
ax.set_xlim(0.5,float(nexp)-0.5)

ax.set_xticklabels(titles[:], rotation= 30, ha="right")
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.25)

fig.savefig('png/fig13.png')
#
