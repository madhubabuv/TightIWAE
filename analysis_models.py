from __future__ import print_function
import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import numpy as np
import pdb

import pytorch_ssim

# model_load_path = "../../___PLOTTING/TightIWAE/___PLOTTING/1_reference_model/logs/model_MIVAE_M8_k8_repeat10_best.pt"
log_model_path = "../../___PLOTTING/TightIWAE/___PLOTTING/1_reference_model/logs/"
model_names = ["model_CIVAE_beta0.5_repeat10_best.pt", #CIWAE
               "model_MIVAE_M1_k1_repeat10_best.pt",   #VAE
               "model_MIVAE_M1_k64_repeat10_best.pt",  #IWAE
               "model_MIVAE_M8_k8_repeat10_best.pt",   #MIWAE
               "model_PIWAE_M8_k8_repeat10_best.pt"]   #PIWAE 
model_load_paths = [log_model_path + model_name for model_name in model_names]

model_hidden_size = 200
model_latent_size = 50
model_num_layers = 2

# =================================

import argparse
import torch
import torch.utils.data
import numpy as np
from utils import *
from model import VAE

parser = argparse.ArgumentParser(description='IWAE Model Analysis')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
#parser.add_argument('--dataset_name', type=str, default='mnist', metavar='DN',
#                    help='name of the dataset: mnist, omniglot')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print("runnning on", device)

#args.batch_size = 8*8 # grid size
#args.dataset_name = 'omniglot'

if args.dataset_name == 'mnist':
    from datasets import load_binarised_MNIST

    path = "./datasets/MNIST"
    train_loader, test_loader, input_size = load_binarised_MNIST(path, args.cuda, args.batch_size)
elif args.dataset_name == 'omniglot':
    from datasets import load_OMNIGLOT

    path = "./datasets/omniglot"
    train_loader, test_loader, input_size = load_OMNIGLOT(path, args.cuda, args.batch_size)

model = VAE(input_size=input_size, device=device, hidden_size=model_hidden_size, latent_size=model_latent_size, num_layers=model_num_layers).to(device)


def normalized_effect_sample_size(data, M, k):

    importance_weights = model.sample_importance_weights_w_i(data, M, k)
    Z = np.sum(importance_weights)
    normalized_importance_weights = (importance_weights/Z)
    ESS = (np.sum(normalized_importance_weights))**2 / np.sum(normalized_effect_sample_size**2)
    
    return ESS

def analyse_model(data, M=8, k=8):
    
    latent_activities = []
    for data, _ in test_loader:
        
        q_z_given_x = model.sample_q_z_given_x(data, M=8, k=8)
        q_z_given_x = q_z_given_x.view(batch_size, model.latent,size, M*k)

        for latent_i in range(q_x_given_x_means.shape[-1]):
            latent_i_data = q_z_given_x[:,latent_i,:]
            latent_i_means = np.mean(latent_i_data, axis = 2, keepdims = False)
            activity_latent_i = np.var(latent_i_means, axis = 1, keepdims = False)
            latent_activities.append(activity_latent_i)
            
        ESS = normalized_effect_sample_size(importance_weights)
        break
    return latent_activities, ESS

if __name__ == "__main__":
    
    models_latent_activities = []
    models_ESS = []    
    for model_load_path in model_load_paths:
        
        model = torch.load(model_load_path)
        model.to(device)
        model.eval()
        
        model_latent_activities, model_ESS = analyse_model(data, M=8, k=8)
        models_latent_activities.append(model_latent_activities)
        models_ESS.append(model_ESS)