#------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------
### config
nexp=8

tfile='./data/loss_p6.txt'

dadir=('nocorr','linear','linear_4','Dense_single','LSTM')

lstyles = ('solid','dotted','dashdot',(0,(3,5,1,5)),(0,(3,2,1,2)))


#------------------------------------------------



with open(tfile) as f:
  lines=f.readlines()

nexp=len(lines)

titles=[]
values=[]
colors=[]

titles.append('')

for i in range(len(lines)) : 
  aline=lines[i].split(' ')
  titles.append(aline[0])
 
  print(i,aline[1].rstrip())
  values.append(float(aline[1].rstrip()))

  if np.mod(i,2) == 0 : 
    colors.append('black')
  else:
    colors.append('grey')

print(np.arange(float(nexp)))
print(np.array(values))
print(np.array(titles))


fig, ax = plt.subplots()

ax.bar(np.arange(float(nexp)), np.array(values), width=0.5, color=colors)
#plt.axhline(y=0, color='k', linestyle='--')
#plt.xlabel('forecast',fontsize=12)
ax.set_ylabel('Validation loss',fontsize=12)

ax.set_ylim(0.0,0.25)
ax.set_xlim(-0.5,float(nexp)-0.5)

ax.set_xticklabels(titles, rotation= 30, ha="right")
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.25)
#plt.xlim()
#plt.ylim()

fig.savefig('png/fig6.png')
#
