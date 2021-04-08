from pylab import plt
from utils import *
import numpy as np

#log_name = "logs/log_MIVAE_M4_k16.h5"


#log_name = "logs/log_MIVAE_M8_k8.h5"
log_name = "logs/CIVAE_beta_record"

#metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_name)
#metrics_iwae_64 = np.asarray(metrics_iwae_64) * -1

### Figure 5 reconstruction

fp = open(log_name,'r')
lines = fp.readlines()
fp.close()
beta_values = []
for idx in range(len(lines)):

    line = lines[idx]
    line = line.split(',')
    
    beta_values.append(float(line[1]))
    
beta_values = np.array(beta_values)
plt.figure()
plt.plot(beta_values, label = "Beta",linewidth=2)

#plt.legend()
plt.ylim([0,1])
plt.grid()
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Beta',fontsize=10)
plt.show()
