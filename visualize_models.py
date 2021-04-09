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

model_load_path = "../../___PLOTTING/TightIWAE/___PLOTTING/1_reference_model/logs/model_MIVAE_M8_k8_repeat10_best.pt"
log_load_path = "../../___PLOTTING/TightIWAE/___PLOTTING/1_reference_model/logs/log_MIVAE_M8_k8_repeat10.h5"
M = 8
k = 8

# =================================

import argparse
import torch
import torch.utils.data
import numpy as np
from utils import *
from model import VAE

parser = argparse.ArgumentParser(description='VAE MNIST Visualizer')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset_name', type=str, default='mnist', metavar='DN',
                    help='name of the dataset: mnist, omniglot')

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


def debug_shape(item):
    return item.cpu().detach().numpy().shape

model = VAE(input_size=input_size, device=device, hidden_size=200, latent_size=50, num_layers=2).to(device)
# use the same details from the original IWAE paper
# hidden size 200
# latent size 50
# number of layers in Enc/Dec only 2

def visualize(viz_name):
    model.eval()
    image_shape = (28, 28)
    ssim_loss = pytorch_ssim.SSIM()

    with torch.no_grad():

        for data, _ in test_loader:
            reconstruction, elbo, loss = model(data, M=1, k=1)
            samples_n = len(data) # 20

            out_grid = reconstruction.view(samples_n, 1, 28, 28)
            in_grid = data.view(samples_n, 1, 28, 28)

            print("in_grid.shape",in_grid.shape)
            print("out_grid.shape",out_grid.shape)

            save_image(in_grid, 'inputs_'+viz_name+'.png')
            save_image(out_grid, 'reconstructions_'+viz_name+'.png')
            ssim_losses = []
            for item_idx in range(samples_n):
                in_data = in_grid[item_idx,0].reshape((1,1,)+image_shape)
                out_data = out_grid[item_idx,0].reshape((1,1,)+image_shape)

                in_data, out_data = in_data.cuda(), out_data.cuda()

                #print("in_data", in_data.shape)
                #print("out_data", out_data.shape)

                ssim = ssim_loss(in_data, out_data)
                ssim_losses.append(ssim.cpu().numpy())

            ssim_losses = np.asarray(ssim_losses)
            print("average ssim loss:", np.mean(ssim_losses), np.std(ssim_losses))

            break


if __name__ == "__main__":
    metrics_iwae_k = []
    metrics_iwae_64 = []
    metrics_iwae_5000 = []

    # model = torch.load("logs/model_" + files_name + "_best.pt")
    model = torch.load(model_load_path)
    model.to(device)
    model.eval()

    metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_load_path)
    start_epoch = len(metrics_iwae_k)
    print("Loading model from ", model_load_path, "logs from", log_load_path, "and skipping to epoch", start_epoch)


    viz_name = args.dataset_name + "_" + model_load_path.split("/")[-1]
    print(model)
    visualize(viz_name)