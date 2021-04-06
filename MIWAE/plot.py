from pylab import plt
from utils import *
import numpy as np
import pandas as pd

piwaes = ["../log_PIWAE_M8_k8.h5", "../log_PIWAE_M8_k8_repeat02.h5"]
miwaes = ["../log_MIVAE_M8_k8.h5", "../log_MIVAE_M8_k8_repeat02.h5"]
#miwae_no_sched = "../_oldlogs/log_MIVAE_M8_k8--note-without-scheduling.h5"
iwaes = ["../log_MIVAE_M1_k64.h5", "../log_MIVAE_M1_k64_repeat02.h5", ]
vaes = ["../log_MIVAE_M1_k1.h5", "../log_MIVAE_M1_k1_repeat02.h5", ]
ciwaes = ["../log_CIVAE_beta0.5.h5", "../log_CIVAE_beta0.5_repeat02.h5", ]

### Figure 5 reconstruction
select_iwae_64 = lambda x: x[1]
select_iwae_5000 = lambda x: x[2]

plt.figure(figsize=(16,4))
plt.title("Figure 5.a replication, IWAE64")
select_func = select_iwae_64
rolling_mean = 5
#rolling_mean = None


plt.title("Figure 5.b replication, log^p(x) (= IWAE5000)")
select_func = select_iwae_5000
rolling_mean = None

def plot_h5_file_k64(log_name, label, color):
    metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_name)
    selected = select_func([metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000])
    plt.plot(np.asarray(selected) * -1, label=label, color=color)

def plot_h5_file_k64_average(log_names, label, color, skip_every = 0):
    repetitions = []
    for log_name in log_names:
        metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_name)
        selected = select_func([metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000])

        repetitions.append(selected)

    number_of_epochs_max = 0
    number_of_epochs_min = 999999
    for arr in repetitions:
        number_of_epochs_max = max(len(arr), number_of_epochs_max)
        number_of_epochs_min = min(len(arr), number_of_epochs_max)

    average_values = []
    std_values = []

    xs = []
    counter = 0
    for epoch_idx in range(number_of_epochs_max):
        if counter >= skip_every:
            counter = 0
        else:
            counter += 1
            continue

        xs.append(epoch_idx)
        values = []
        for arr_idx in range(len(repetitions)):
            arr = repetitions[arr_idx]
            if epoch_idx < len(arr):
                #print(len(arr), epoch_idx)
                values.append(arr[epoch_idx])

        values = np.asarray(values)
        #print(values)
        average_values.append(np.mean(values))
        std_values.append(np.std(values))

    print("from epochs 0 to", number_of_epochs_min," averages, then only one value until", len(std_values))

    average_values = -1 * np.asarray(average_values)
    std_values = np.asarray(std_values)

    if rolling_mean is not None:
        pd_average_values = pd.DataFrame(average_values)
        p = plt.plot(pd_average_values[0].rolling(rolling_mean).mean(), color, label=label)
        #p = plt.plot(pd_average_values[0], color, pd_average_values[0].rolling(rolling_mean).mean(), color, label=label)

    else:
        p = plt.plot(xs, average_values, label=label, color=color)
        col = p[0].get_color()

        xs = list(range(len(average_values)))
        plt.fill_between(xs, average_values-std_values, average_values+std_values, color=col, alpha=0.25)




# MIVAE M8, k8
plot_h5_file_k64_average(piwaes, "PIWAE M=8, k=8", color='#c44e52')
plot_h5_file_k64_average(miwaes, "MIWAE M=8, k=8", color='#8172b2')
#plot_h5_file_k64(miwae_no_sched, "MIWAE M=8, k=8, no sched")
plot_h5_file_k64_average(iwaes, "IWAE k=64", color='#4c72b0')
plot_h5_file_k64_average(vaes, "VAE", color='#ccb974')
plot_h5_file_k64_average(ciwaes, "CIWAE beta=0.5", color='#55a868')

#plt.ylim(-92,-83)
plt.ylim(-94.5,-82)
plt.xlim(0,2700)


plt.legend()
plt.show()