# Based on implementations
# - vae core https://github.com/pytorch/examples/blob/master/vae/main.py
# - miwae https://github.com/yoonholee/pytorch-vae

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from load_data import load_dataset
import numpy as np
import pdb
from utils import *
from model import VAE

#from datasets import load_binarised_MNIST

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--test_batch_size', type=int, default=20, metavar='BStest',
                    help='input batch size for testing (default: 20)')
parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

parser.add_argument('--dataset_name', type=str, default='freyfaces', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes, histopathologyGray, freyfaces, cifar10')

parser.add_argument('--hidden_units', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 200)')

#parser.add_argument('--dataset_dir', default="./datasets/MNIST", 
#                    help='dataset directory')

parser.add_argument('--epochs', type=int, default=3280, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
#                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=float, default=0.5, help='beta for CIWAE')
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--M', type=int, default=1)
parser.add_argument('--piwae', action='store_true', default=False)
parser.add_argument('--miwae', action='store_true', default=False)
parser.add_argument('--ciwae', action='store_true', default=False)
parser.add_argument('--cont', action='store_true', default=False)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.log_interval = 1
#torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
piwae = args.piwae
miwae = args.miwae
ciwae = args.ciwae

print("runnning on", device)
if piwae:
    print('Using PIWAE\n')
elif miwae:
    print('Using MIWAE\n')
elif ciwae:
    beta = args.beta
    print('Using CIWAE with beta ' +str(beta)+'\n' )


#train_loader, test_loader, input_size = load_binarised_MNIST(args.dataset_dir, args.cuda, args.batch_size)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

input_size = args.input_size[1] * args.input_size[2]


if 'cifar' in args.dataset_name:
    input_size*=3

def debug_shape(item):
    return item.cpu().detach().numpy().shape

args.log_interval = 500
# learning rate over epochs
epoch_num = 0
milestones = []
for i in range(8):
    epoch_num += 3**i
    milestones.append(epoch_num)

model = VAE(input_size=input_size, hidden_size = args.hidden_units, latent_size=50, piwae=piwae,device=device,input_type = args.input_type).to(device)

if piwae:
    optimizer_encoder = optim.Adam(model.encoder.parameters(),lr=1e-3)
    optimizer_decoder = optim.Adam(model.decoder.parameters(),lr=1e-3)
    scheduler_enc = optim.lr_scheduler.MultiStepLR(optimizer_encoder, milestones=milestones, gamma=10 ** (-1 / 7))
    scheduler_dec = optim.lr_scheduler.MultiStepLR(optimizer_decoder, milestones=milestones, gamma=10 ** (-1 / 7))

else:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=10 ** (-1 / 7))


def train(epoch,M,k):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        if piwae:
            
            _, elbo, loss_mk = model(data, M=10, k=10)
            encoder_loss = loss_mk.mean()
            optimizer_encoder.zero_grad()
            encoder_loss.backward()
            optimizer_encoder.step()
            
            _, elbo, loss_mk = model(data, M=1, k=10)
            decoder_loss = loss_mk.mean()
            optimizer_decoder.zero_grad()
            decoder_loss.backward()
            optimizer_decoder.step()
            
            
            loss = (encoder_loss.item() + decoder_loss.item())/2.0
            train_loss += loss
        elif miwae:
            _, elbo, loss_mk = model(data, M, k)
            loss = loss_mk.mean()
            optimizer.zero_grad()
            loss.backward()
    
            train_loss += loss.item()
            optimizer.step()
            loss = loss.item()
        elif ciwae:
            
            ###ELBO_VAE
            M = 1### M > 1 only in MIWAE
            _, elbo, loss_mk = model(data, M, k=1)
            loss_VAE = loss_mk.mean()
            
            ###ELBO_IWAE
            _, elbo, loss_mk = model(data, M, k)
            loss_IWAE = loss_mk.mean()
            
            loss = loss_VAE*beta + loss_IWAE*(1-beta)
            
            optimizer.zero_grad()
            loss.backward()
    
            train_loss += loss.item()
            optimizer.step()
            loss = loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss )) # / len(data)


def test(epoch):
    #print_metrics = ((epoch-1) % 10) == 0
    print_metrics = True
    if print_metrics:
        model.eval()

        with torch.no_grad():
            # Tests:
            # IWAE with k, IWAE with 64, IWAE with 5000

            elbos = []
            for data, _ in test_loader:
                _, elbo, _ = model(data, M=1, k=5000)
                elbos.append(elbo.squeeze(0))
            elbos = np.asarray(elbos)

            k_to_run = [k, 64, 5000]
            all_losses = []
            for k_for_loss in k_to_run:
                losses = []
                for elbo in elbos[:k_for_loss]:
                    losses.append(model.logmeanexp(elbo, 0).cpu().numpy().flatten())

                loss = np.concatenate(losses).mean()

                all_losses.append(- loss)
            test_loss_iwae_k, test_loss_iwae64, test_loss_iwae5000 = all_losses

            print('====>Test metrics: IWAE M=', M, ',k=',k, ' || epoch', epoch)
            print("IWAE-64: ", test_loss_iwae64)
            print("logˆp(x) = IWAE-5000: ", test_loss_iwae5000)
            print("−KL(Q||P): ", test_loss_iwae64-test_loss_iwae5000)
            print("---------------")

            return test_loss_iwae_k, test_loss_iwae64, test_loss_iwae5000





if __name__ == "__main__":
    metrics_iwae_k = []
    metrics_iwae_64 = []
    metrics_iwae_5000 = []

    M = args.M
    k = args.k

    if miwae:
        files_name = "MIVAE_M"+str(M)+"_k"+str(k)
    if piwae:
        files_name = "PIWAE_M"+str(M)+"_k"+str(k)
    if ciwae:
        files_name = "CIVAE_beta"+str(beta)
    files_name = files_name + "_" + args.dataset_name
    log_name = "logs/log_" + files_name + ".h5"
    model_name = "logs/model_" + files_name + ".pt"


    if args.cont:
        #model = torch.load("logs/model_" + files_name + "_best.pt")
        model = torch.load(model_name)
        model.to(device)
        model.eval()

        metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000 = load_from_h5(log_name)
        start_epoch = len(metrics_iwae_k)
        print("Loading model from ", model_name, "logs from", log_name, "and skipping to epoch", start_epoch)

    mkdir("results")
    mkdir("logs")

    best_test_loss = np.inf
    
    try:
        for epoch in range(1, args.epochs + 1):
            if args.cont:
                if epoch < start_epoch:
                    continue

            train(epoch,M,k)
            test_loss_iwae_k, test_loss_iwae64, test_loss_iwae5000 = test(epoch)
            metrics_iwae_k.append(test_loss_iwae_k)
            metrics_iwae_64.append(test_loss_iwae64)
            metrics_iwae_5000.append(test_loss_iwae5000)

            save_to_h5(metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000, log_name)
            torch.save(model, model_name)

            if test_loss_iwae_k < best_test_loss: # minimizing loss
                best_test_loss = test_loss_iwae_k
                torch.save(model, "logs/model_" + files_name + "_best.pt")

            #with torch.no_grad():
            #    sample = torch.randn(64, 20).to(device)
            #    sample = model.decode(sample).probs.cpu() # piwae AttributeError: 'VAE' object has no attribute 'decode'

            #    save_image(sample.view(64, 1, 28, 28), 'results/sample_epoch' + str(epoch).zfill(4) + '.png')

            if piwae:
                scheduler_enc.step()
                scheduler_dec.step()
            else:
                scheduler.step()

    except KeyboardInterrupt as E:

        print("Interrupted! Ended the run with exception", E)
        print("metrics_iwae_k = ", metrics_iwae_k)
        print("metrics_iwae_64 = ", metrics_iwae_64)
        print("metrics_iwae_5000 = ", metrics_iwae_5000)
    
