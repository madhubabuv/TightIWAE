import h5py
import numpy as np
import pandas as pd
from pylab import plt

def load_from_h5(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    metrics_iwae_k = hdf5_file['metrics_iwae_k'][:]
    metrics_iwae_64 = hdf5_file['metrics_iwae_64'][:]
    metrics_iwae_5000 = hdf5_file['metrics_iwae_5000'][:]
    hdf5_file.close()
    return list(metrics_iwae_k), list(metrics_iwae_64), list(metrics_iwae_5000)

def plot_h5_file_k64(log_name, label, color, f):
    metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_name)
    selected = f([metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000])
    plt.plot(np.asarray(selected) * -1, label=label, color=color)

def plot_h5_file_k64_average(log_names, label, color, skip_every = 0, at = None, select_func=None, rolling_mean=None):
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

    #print("from epochs 0 to", number_of_epochs_min," averages, then only one value until", len(std_values))

    average_values = -1 * np.asarray(average_values)
    std_values = np.asarray(std_values)
    if rolling_mean is not None:
        pd_average_values = pd.DataFrame(average_values)
        p = plt.plot(pd_average_values[0].rolling(rolling_mean).mean(), color, label=label)

    else:
        p = plt.plot(xs, average_values, label=label, color=color)
        col = p[0].get_color()

        xs = list(range(len(average_values)))
        plt.fill_between(xs, average_values-std_values, average_values+std_values, color=col, alpha=0.25)

    if at is not None:
        #print("sampling at", at-20, "to", at)
        last_20_avg_values = average_values[at-20:at]
        return np.mean(last_20_avg_values)

    else:
        last_20_avg_values = average_values[-20:]
        return np.mean(last_20_avg_values)

def min_epoch(arrays, select_func):
    available = []
    for log_names in arrays:
        for log_name in log_names:
            metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_name)
            selected = select_func([metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000])

            #print("this one has", len(selected))
            available.append(len(selected))
    return min(available)
