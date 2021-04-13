#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --job-name=TightIWAE
#SBATCH --partition=htc
#SBATCH --gres=gpu:1 --constraint='gpu_sku:K80|gpu_sku:P100|gpu_sku:V100'

# Note that the exact versions loaded modules and conda environments will depend on your setup and your Cluster...
module load anaconda3/2019.03
module load gpu/cuda/10.0.130
module load gpu/cudnn/7.5.0__cuda-10.0

#(first create and prepare the environment)
source activate $DATA/bayesian-env
#(first run) git clone https://github.com/madhubabuv/TightIWAE.git
cd $DATA/python_codes/TightIWAE/

nvidia-smi

##==-- Experiments to run: --==##
# PIWAE (M=8,k=8)
python miwae_simplified.py --piwae --k 8 --M 8 --dataset_name mnist

# MIWAE (M=8,k=8)
python miwae_simplified.py --miwae --k 8 --M 8 --dataset_name mnist

# IWAE (M=1,k=64)
python miwae_simplified.py --miwae --k 64 --M 1 --dataset_name mnist

# CIWAE (Beta=0.5)
python miwae_simplified.py --ciwae --beta 0.5 --dataset_name mnist

# VAE (M=1,k=1)
python miwae_simplified.py --miwae --k 1 --M 1 --dataset_name mnist

echo "Finished!"
