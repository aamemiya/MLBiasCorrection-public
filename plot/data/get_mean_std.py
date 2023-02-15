import numpy as np
import sys

cfile=sys.argv[1]
with open(cfile,"r") as f:
  vals=f.readlines()
  for j in range(len(vals)):
     vals[j]=float(vals[j])

vmean=np.mean(vals)
vstd=np.std(vals)

print(vmean,vstd)

