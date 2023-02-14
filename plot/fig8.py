#------------------------------------------------
import os
import sys
import numpy as np
#import param
import matplotlib.pyplot as plt

#------------------------------------------------
### config
nx=8

exp = 'p8'

items = ['nocorr','linear','linear4','Dense_single','LSTM']
#items = ['nocorr','linear','linear4']
legends = ['None','Linear','Linear4th','Dense_single','LSTM']
#legends = ['None','Linear','Linear4th']
lstyles = ['solid','dotted',(0,(1,4)),'dashed','dashdot','dashed','dashdot']
#lstyles = ['solid','dotted',(0,(1,4))]

#------------------------------------------------

nitem=len(items)

infl=[]
rmse=[]

# load nature and observation data
for i in range(nitem):
  infl.append([])
  rmse.append([])
  fname='./data/log_sweep_' + exp + '_' + items[i] 
  with open(fname,'r') as f:
    line=f.readlines()
    length=len(line)
    for j in range(length): 
      infl[i].append(float(line[j].split()[1]))
      rmse[i].append(float(line[j].split()[3]))

###
plt.figure()
#plt.yscale('log')
for i in range(nitem):
  plt.plot(infl[i], rmse[i],label=legends[i],linestyle=lstyles[i])

plt.legend(bbox_to_anchor=(0.95,0.95), loc='upper right', borderaxespad=0,fontsize=14)
plt.xlabel('Additive inflation factor')
plt.ylabel('Analysis RMSE')
#plt.yticks(ticks=(0.1,0.2,0.5,1.0,2.0),labels=("0.1","0.2","0.5","1.0","2.0"))
plt.yticks(ticks=(0.0,0.1,0.2,0.3),labels=("0.0","0.1","0.2","0.3"))
#plt.yticks(ticks=(0.0,0.05,0.1,0.15,0.2),labels=("0.0","0.05","0.1","0.15","0.2"))
#plt.yticklabels()
#plt.axis('equal')
#plt.axis('square')
plt.xlim(0.0,1.1)
#plt.ylim(0.05,2.0)
#plt.ylim(0.0,2.0)
plt.ylim(0.0,0.3)
plt.savefig('png/fig8.png')
###

