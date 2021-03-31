import os
import h5py

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# === specific load and saves:
def save_to_h5(metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000, path):
    directory = os.path.dirname(path)
    mkdir(directory)

    hdf5_file = h5py.File(path, mode='w')
    hdf5_file.create_dataset("metrics_iwae_k", data=metrics_iwae_k, dtype="float32")
    hdf5_file.create_dataset("metrics_iwae_64", data=metrics_iwae_64, dtype="float32")
    hdf5_file.create_dataset("metrics_iwae_5000", data=metrics_iwae_5000, dtype="float32")
    hdf5_file.close()


def load_from_h5(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    metrics_iwae_k = hdf5_file['metrics_iwae_k'][:]
    metrics_iwae_64 = hdf5_file['metrics_iwae_64'][:]
    metrics_iwae_5000 = hdf5_file['metrics_iwae_5000'][:]
    hdf5_file.close()
    return list(metrics_iwae_k), list(metrics_iwae_64), list(metrics_iwae_5000)
