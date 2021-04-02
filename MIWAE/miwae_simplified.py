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
import numpy as np

from utils import *

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=3280, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--k', type=int, default=1)
parser.add_argument('--M', type=int, default=1)

parser.add_argument('--cont', action='store_true', default=False)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.log_interval = 1
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
print("runnning on", device)

from datasets import load_binarised_MNIST

path = "./datasets/MNIST"
train_loader, test_loader, input_size = load_binarised_MNIST(path, args.cuda, args.batch_size)

def debug_shape(item):
    return item.cpu().detach().numpy().shape

class VAE(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 400, latent_size = 20):
        super(VAE, self).__init__()

        # encoder layers
        self.fc11 = nn.Linear(input_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        # decoder layers
        self.fc31 = nn.Linear(latent_size, hidden_size)
        self.fc32 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.input_size = input_size

        self.prior_distribution = Normal(torch.zeros([self.latent_size]).to(device), torch.ones([self.latent_size]).to(device))

    def encode(self, x):
        x = torch.tanh(self.fc11(x))
        x = torch.tanh(self.fc12(x))

        mu_enc = self.fc21(x)
        std_enc = self.fc22(x)
        return Normal(mu_enc, F.softplus(std_enc))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.tanh(self.fc31(z))
        x = F.tanh(self.fc32(x))
        x = self.fc4(x)
        return Bernoulli(logits=x)
        #return ContinuousBernoulli(logits=x) # note that MNIST is still binarized ...
        # by default should be Bernoulli (for binarized MNIST), but see:
        # The continuous Bernoulli: fixing a pervasive error in variational autoencoders https://arxiv.org/abs/1907.06845

    def forward(self, x, M, k):
        input_x = x.view(-1, self.input_size).to(device)
        # encoded distribution ~ q(z|x, params) = Normal (real input_x; encoder_into_Mu, encoder_into_Std )
        z_distribution = self.encode(input_x)
        # sample z values from this distribution
        z = z_distribution.rsample(torch.Size([M, k]))

        # reconstructions distribution ~ p(x|z, params) = Normal/Bernoulli (sampled z)
        x_distribution = self.decode(z)

        # priors distribution ~ p(z) = Normal (sampled z; 0s, 1s ) 
        #self.prior_distribution = Normal(torch.zeros([self.latent_size]).to(device), torch.ones([self.latent_size]).to(device))

        elbo = self.elbo(input_x, z, x_distribution, z_distribution)  # mean_n, imp_n, batch_size
        elbo_iwae = self.logmeanexp(elbo, 1).squeeze(1)  # mean_n, batch_size
        loss = - torch.mean(elbo_iwae, 0)  # batch_size

        return x_distribution.probs, elbo, loss

    def logmeanexp(self, inputs, dim=1): # ***
        if inputs.size(dim) == 1:
            return inputs
        else:
            input_max = inputs.max(dim, keepdim=True)[0]
            return (inputs - input_max).exp().mean(dim).log() + input_max

    def elbo(self, input_x, z, x_distribution, z_distribution):
        lpxz = x_distribution.log_prob(input_x).sum(-1)

        lpz = self.prior_distribution.log_prob(z).sum(-1)
        lqzx = z_distribution.log_prob(z).sum(-1)
        kl = -lpz + lqzx
        return -kl + lpxz


args.log_interval = 500
M = args.M
k = args.k
#M = 5
#k = 5

# learning rate over epochs
epoch_num = 0
milestones = []
for i in range(8):
    epoch_num += 3**i
    milestones.append(epoch_num)

model = VAE(input_size=input_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=10 ** (-1 / 7))

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        _, elbo, loss_mk = model(data, M, k)
        loss = loss_mk.mean()

        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() )) # / len(data)


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

    files_name = "MIVAE_M"+str(M)+"_k"+str(k)
    log_name = "logs/log_" + files_name + ".h5"
    model_name = "logs/model_"+files_name+".pt"

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

            train(epoch)
            test_loss_iwae_k, test_loss_iwae64, test_loss_iwae5000 = test(epoch)
            metrics_iwae_k.append(test_loss_iwae_k)
            metrics_iwae_64.append(test_loss_iwae64)
            metrics_iwae_5000.append(test_loss_iwae5000)

            save_to_h5(metrics_iwae_k, metrics_iwae_64, metrics_iwae_5000, log_name)
            torch.save(model, model_name)

            scheduler.step()

            if test_loss_iwae_k < best_test_loss: # minimizing loss
                best_test_loss = test_loss_iwae_k
                torch.save(model, "logs/model_" + files_name + "_best.pt")

            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).probs.cpu()

                save_image(sample.view(64, 1, 28, 28), 'results/sample_epoch' + str(epoch).zfill(4) + '.png')

    except KeyboardInterrupt as E:

        print("Interrupted! Ended the run with exception", E)
        print("metrics_iwae_k = ", metrics_iwae_k)
        print("metrics_iwae_64 = ", metrics_iwae_64)
        print("metrics_iwae_5000 = ", metrics_iwae_5000)