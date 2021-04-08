from pylab import plt
from utils import *
import numpy as np

#log_name = "logs/log_MIVAE_M4_k16.h5"

M = 8
k = 8
files_name = "MIVAE_M" + str(M) + "_k" + str(k)
log_name = "logs/log_" + files_name + ".h5"

#log_name = "logs/log_MIVAE_M8_k8.h5"

metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_name)
metrics_iwae_64 = np.asarray(metrics_iwae_64) * -1

### Figure 5 reconstruction

plt.figure()
plt.plot(metrics_iwae_64, label = "MIWAE M=4, k=16")

plt.legend()
plt.show()