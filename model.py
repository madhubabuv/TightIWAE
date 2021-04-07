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

min_epsilon = 1e-5
max_epsilon = 1.-1e-5

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
    def __init__(self, input_size, hidden_size, latent_size, input_type = 'binary'):
        super(Decoder, self).__init__()
        self.fc31 = nn.Linear(latent_size, hidden_size)
        self.fc32 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.input_type = input_type
        if self.input_type != 'binary':
            self.fc5= nn.Linear(hidden_size, input_size)
        
    def forward(self,x,z):
        x = F.tanh(self.fc31(z))
        x = F.tanh(self.fc32(x))

        if self.input_type == 'binary':
            mean = self.fc4(x)
            mean = F.sigmoid(mean)
            return Bernoulli(logits=mean),None
        else:
            mean = self.fc4(x)
            mean = F.sigmoid(mean)
            std_dec = self.fc5(x)
            return Normal(mean,F.softplus(std_dec))

class VAE(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 200, latent_size = 20, piwae=False, device=torch.device('cuda'), input_type = 'binary'):
        super(VAE, self).__init__()

        self.piwae = piwae
        self.device = device
        self.input_type = input_type
        
        # encoder layers
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        
        # decoder layers
        self.decoder = Decoder(input_size, hidden_size, latent_size,input_type=input_type)
        

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

            if self.input_type == 'binary':

                #print("Input image type to be found Binary, so going with Bernoulli")

                x_distribution = self.decoder(x,z,bernoulli=True)
            else:

                #print('input image type to be found, ',self.input_type," so,using Normal distribution")

                x_distribution = self.decoder(x,z)
    
            elbo = self.elbo(input_x, z, x_distribution, z_distribution)  # mean_n, imp_n, batch_size
            elbo_iwae = self.logmeanexp(elbo, 1).squeeze(1)  # mean_n, batch_size
            loss = -1.* torch.mean(elbo_iwae, 0)  # batch_size

            return x_distribution, elbo, loss 
                
            
                
        else:
            input_x = x.view(-1, self.input_size).to(device)
            # encoded distribution ~ q(z|x, params) = Normal (real input_x; encoder_into_Mu, encoder_into_Std )
            
            z_distribution = self.encoder(input_x)
            
            # sample z values from this distribution
            z = z_distribution.rsample(torch.Size([M, k]))
    
            # reconstructions distribution ~ p(x|z, params) = Normal/Bernoulli (sampled z)
            if args.input_type =='binary':

                x_distribution = self.decoder(z,bernoulli=True)

            else:

                x_distribution = self.decoder(z)
    
            # priors distribution ~ p(z) = Normal (sampled z; 0s, 1s ) 
            #self.prior_distribution = Normal(torch.zeros([self.latent_size]).to(device), torch.ones([self.latent_size]).to(device))
    
            elbo = self.elbo(input_x, z, x_distribution, z_distribution)  # mean_n, imp_n, batch_size

            elbo_iwae = self.logmeanexp(elbo, 1).squeeze(1)  # mean_n, batch_size

            loss = -torch.mean(elbo_iwae, 0)  # batch_size
            
            return x_distribution, elbo, loss

    def logmeanexp(self, inputs, dim=1): # ***
        if inputs.size(dim) == 1:
            return inputs
        else:
            input_max = inputs.max(dim, keepdim=True)[0]
            return (inputs - input_max).exp().mean(dim).log() + input_max

    def elbo(self, input_x, z, x_distribution, z_distribution):

        if self.input_type == 'binary':

            RE  = self.log_Bernoulli(input_x, x_distribution.mean, dim=-1)

        elif self.input_type == 'gray' or self.input_type == 'continuous':

            RE = -1. * self.log_Logistic_256(input_x, x_distribution.loc, logvar=x_distribution.scale , dim=-1)

            RE = -1. * self.log_Logistic_256(input_x, x_distribution.mean, logvar=torch.tensor(0.0), dim=1)
            RE = torch.sum(RE, dim=2) # sum reconstruction err (1, 20, 784) > (1, 20)
            RE = RE / self.input_size # divided by number of pixels > mean reconstruction err

        else:

            raise Exception('Wrong input type!')

        lpz = self.prior_distribution.log_prob(z).sum(-1)
        
        lqzx = z_distribution.log_prob(z).sum(-1)

        kl = -1. * lpz + lqzx

        return -1. * kl + RE


    def log_Bernoulli(self, x, mean, average=False, dim=None):

        probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon )

        log_bernoulli = x * torch.log( probs ) + (1. - x ) * torch.log( 1. - probs )

        if average:
            return torch.mean( log_bernoulli, dim )
        else:
            return torch.sum( log_bernoulli, dim )

    def logisticCDF(self,x, u, s):
        return 1. / ( 1. + torch.exp( -(x-u) / s ) )

    def sigmoid(self,x):
        return 1. / ( 1. + torch.exp( -x ) )

    def log_Logistic_256(self,x, mean, logvar, average=False, reduce=True, dim=None):
        bin_size = 1. / 256.

        # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
        scale = logvar #torch.exp(logvar)
        x = (torch.floor(x / bin_size) * bin_size - mean) / scale

        cdf_plus = torch.sigmoid(x + bin_size/scale)
        cdf_minus = torch.sigmoid(x)

        # calculate final log-likelihood for an image
        log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

        if reduce:
            if average:
                return torch.mean(log_logist_256, dim)
            else:
                return torch.sum(log_logist_256, dim)
        else:
            return log_logist_256


        