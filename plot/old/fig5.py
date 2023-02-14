#------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------
### config

tfile='./data/loss_p8.txt'

#------------------------------------------------



with open(tfile) as f:
  lines=f.readlines()

nexp=len(lines)

titles=[]
values=[]
errors=[]
colors=[]

titles.append('')

for i in range(len(lines)) : 
  aline=lines[i].split(' ')
  titles.append(aline[0])
 
  print(i,aline[1].rstrip())
  values.append(float(aline[1].rstrip()))

  if (len(aline)==3):
     print(i,aline[2].rstrip())
     errors.append(float(aline[2].rstrip()))
  else:
     errors.append(0.0)

  if np.mod(i,2) == 0 : 
#    colors.append('grey')
    colors.append('black')
  else:
#    colors.append('lightgrey')
    colors.append('grey')

print(np.arange(float(nexp)))
print(np.array(values))
print(np.array(titles))


fig, ax = plt.subplots()

ax.bar(np.arange(float(nexp)), np.array(values), width=0.5, color=colors)
## ax.errorbar(np.arange(float(nexp)), np.array(values), yerr = np.array(errors), capsize=0, ecolor='black', markeredgecolor = "black", linestyle="")

ax.set_ylabel('Validation loss',fontsize=12)

ax.set_ylim(0.0,0.20)
ax.set_xlim(-0.5,float(nexp)-0.5)

ax.set_xticklabels(titles, rotation= 30, ha="right")
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.25)

fig.savefig('png/fig5.png')
#
