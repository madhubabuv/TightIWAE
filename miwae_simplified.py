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
from model import VAE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
#                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=float, default=0.5, help='beta for CIWAE')

parser.add_argument('--k', type=int, default=1)
parser.add_argument('--M', type=int, default=1)
parser.add_argument('--piwae', action='store_true', default=False)
parser.add_argument('--miwae', action='store_true', default=False)
parser.add_argument('--ciwae', action='store_true', default=False)
parser.add_argument('--rpiwae', action='store_true', default=False)

parser.add_argument('--repetition', type=int, default=1)

parser.add_argument('--dataset_name', type=str, default='mnist', metavar='DN',
                    help='name of the dataset: mnist, omniglot')

parser.add_argument('--cont', action='store_true', default=False)
parser.add_argument('--ciwae-beta', action='store_true', default=False)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.log_interval = 1
#torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
piwae = args.piwae
miwae = args.miwae
ciwae = args.ciwae
ciwae_beta = args.ciwae_beta
rpiwae = args.rpiwae

print("runnning on", device)
if piwae:
    print('Using PIWAE\n')
elif miwae:
    print('Using MIWAE\n')
elif ciwae:
    beta = args.beta
    print('Using CIWAE with beta ' +str(beta)+'\n' )
elif ciwae_beta:
    beta = args.beta
    print('Using CIWAE with beta learning, initial beta = ' +str(beta)+'\n' )
elif rpiwae:
    print('Using RPIWAE\n')
    

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

args.log_interval = 500


# learning rate over epochs
epoch_num = 0
milestones = []
for i in range(8):
    epoch_num += 3**i
    milestones.append(epoch_num)

model = VAE(input_size=input_size,piwae=piwae,device=device,
            hidden_size = 200, latent_size = 50, num_layers = 2).to(device)
# use the same details from the original IWAE paper
# hidden size 200
# latent size 50
# number of layers in Enc/Dec only 2

if piwae or rpiwae:
    optimizer_encoder = optim.Adam(model.encoder.parameters(),lr=1e-3)
    optimizer_decoder = optim.Adam(model.decoder.parameters(),lr=1e-3)
    scheduler_enc = optim.lr_scheduler.MultiStepLR(optimizer_encoder, milestones=milestones, gamma=10 ** (-1 / 7))
    scheduler_dec = optim.lr_scheduler.MultiStepLR(optimizer_decoder, milestones=milestones, gamma=10 ** (-1 / 7))
elif ciwae_beta:
    beta = args.beta
    beta = torch.tensor((beta),device="cuda").unsqueeze(0)
    beta = beta.detach()
    beta = beta.requires_grad_()
  
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_beta = optim.Adam([beta], lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=10 ** (-1 / 7))
    scheduler_beta = optim.lr_scheduler.MultiStepLR(optimizer_beta, milestones=milestones, gamma=10 ** (-1 / 7))
else:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=10 ** (-1 / 7))


def train(epoch,M,k):
    model.train()
    train_loss = 0
    beta_record = []
    
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
        elif ciwae_beta:
            
            
            ###Optimizing Beta
            _, elbo, loss_mk = model(data, M, k=1)
            loss_VAE = loss_mk.mean()
            
            ###ELBO_IWAE
            _, elbo, loss_mk = model(data, M, k)
            loss_IWAE = loss_mk.mean()
            
           
            optimizer_beta.zero_grad()
            loss = loss_VAE*beta + loss_IWAE*(1-beta)
            loss.backward()
            optimizer_beta.step()
            
            beta_record.append(beta.item())
            
            
            
            ###ELBO_VAE
            M = 1### M > 1 only in MIWAE
            _, elbo, loss_mk = model(data, M, k=1)
            loss_VAE = loss_mk.mean()
            
            ###ELBO_IWAE
            _, elbo, loss_mk = model(data, M, k)
            loss_IWAE = loss_mk.mean()
            
            loss = loss_VAE*beta.item() + loss_IWAE*(1-beta)
         
            optimizer.zero_grad()
            loss.backward()
    
            train_loss += loss.item()
            optimizer.step()
            loss = loss.item()
        # stochastic order for two target optimization in piwae
        elif rpiwae:
            
            ### first determine order of network updates with Bernoulli trial
            encoder_first = np.random.randint(2)
            
            if encoder_first:
                
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
                
            else:
                
                _, elbo, loss_mk = model(data, M=1, k=10)
                decoder_loss = loss_mk.mean()
                optimizer_decoder.zero_grad()
                decoder_loss.backward()
                optimizer_decoder.step()
                
                _, elbo, loss_mk = model(data, M=10, k=10)
                encoder_loss = loss_mk.mean()
                optimizer_encoder.zero_grad()
                encoder_loss.backward()
                optimizer_encoder.step()
                
            
            loss = (encoder_loss.item() + decoder_loss.item())/2.0
            train_loss += loss
        
            
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),loss )) # / len(data)
    if ciwae_beta:
       return beta_record

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
    if ciwae_beta:
        files_name = "CIVAE_beta_learner"+str(beta.item())
        beta_writer_name = "CIVAE_beta_record"
    if rpiwae:
        files_name = "RPIWAE_M"+str(M)+"_k"+str(k)
    
    if args.dataset_name == 'omniglot':
        files_name = files_name + "_" + args.dataset_name

    if args.repetition != 1:
        files_name += "_repeat"+str(args.repetition).zfill(2) # support for repeated runs on cluster
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
            if ciwae_beta:
                beta_record = train(epoch,M,k)
                beta_writer = open(beta_writer_name,'a+')
                beta_writer.write(str(epoch)+','+str(np.mean(beta_record))+'\n')
                beta_writer.close()
            else:
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
            '''
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decoder(sample).probs.cpu() # piwae AttributeError: 'VAE' object has no attribute 'decode'

                save_image(sample.view(64, 1, 28, 28), 'results/sample_epoch' + str(epoch).zfill(4) + '.png')
            '''
            if piwae or rpiwae:
                scheduler_enc.step()
                scheduler_dec.step()
            elif ciwae_beta:
                scheduler.step()
                scheduler_beta.step()
            else:
                scheduler.step()
        if ciwae_beta:
            beta_writer.close()
    except KeyboardInterrupt as E:

        print("Interrupted! Ended the run with exception", E)
        print("metrics_iwae_k = ", metrics_iwae_k)
        print("metrics_iwae_64 = ", metrics_iwae_64)
        print("metrics_iwae_5000 = ", metrics_iwae_5000)
    
