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
import pdb
from utils import *


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.fc11 = nn.Linear(input_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
    def forward(self,x):
        x = torch.tanh(self.fc11(x))
        x = torch.tanh(self.fc12(x))

        mu_enc = self.fc21(x)
        std_enc = self.fc22(x)
        return Normal(mu_enc, F.softplus(std_enc))
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.fc31 = nn.Linear(latent_size, hidden_size)
        self.fc32 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        
    def forward(self,x,z):
        x = F.tanh(self.fc31(z))
        x = F.tanh(self.fc32(x))
        x = self.fc4(x)
        return Bernoulli(logits=x)
    
class VAE(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 400, latent_size = 20, piwae=False, device=torch.device('cuda')):
        super(VAE, self).__init__()

        self.piwae = piwae
        self.device = device
        
        # encoder layers
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        
        # decoder layers
        self.decoder = Decoder(input_size, hidden_size, latent_size)
        

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.input_size = input_size

        self.prior_distribution = Normal(torch.zeros([self.latent_size]).to(device), torch.ones([self.latent_size]).to(device))
     
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x, M, k,train=True):
        
        
        if train:
            
                
                
                input_x = x.view(-1, self.input_size).to(self.device)
                
                z_distribution = self.encoder(input_x)                                
                z = z_distribution.rsample(torch.Size([M, k]))
        
                
                x_distribution = self.decoder(x,z)
        
                elbo = self.elbo(input_x, z, x_distribution, z_distribution)  # mean_n, imp_n, batch_size
                elbo_iwae = self.logmeanexp(elbo, 1).squeeze(1)  # mean_n, batch_size
                loss = - torch.mean(elbo_iwae, 0)  # batch_size
                return x_distribution.probs, elbo, loss 
                    
            
                
        else:
            input_x = x.view(-1, self.input_size).to(device)
            # encoded distribution ~ q(z|x, params) = Normal (real input_x; encoder_into_Mu, encoder_into_Std )
            
            z_distribution = self.encoder(input_x)
            
            # sample z values from this distribution
            z = z_distribution.rsample(torch.Size([M, k]))
    
            # reconstructions distribution ~ p(x|z, params) = Normal/Bernoulli (sampled z)
            x_distribution = self.decoder(z)
    
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


    