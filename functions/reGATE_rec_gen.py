from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as utils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

###Load data
torch.manual_seed(11111)

data = torch.load('data.pt')
##Create the data loader 
tensor_net = data[0]
tensor_response = data[1]
label= data[2]
y = utils.TensorDataset(tensor_net, tensor_response, label) # create your datset
batch_size = 100
train_loader = utils.DataLoader(y, batch_size,  shuffle=True) 

###Set the loss type:"binary" for binary data; "poisson" for counts data
loss_type="binary"
###Set the latent dimension
latent_dim = 68 

class VAE(nn.Module):
    def __init__(self):
        latent_dim= 68
        encode_dim = 1024
        super(VAE, self).__init__()
        self.fc11 = nn.Linear(68*68,encode_dim)
        self.fc12 = nn.Linear(68*68,encode_dim)
        self.fc21 = nn.Linear(encode_dim,latent_dim)
        self.fc22 = nn.Linear(encode_dim,latent_dim)
        self.drop_layer = nn.Dropout(p=0.5)
        #########
        self.fc31 = nn.Linear(latent_dim,68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)
        #########
        self.fc41 = nn.Linear(68,68)
        self.fc42 = nn.Linear(68,68)
        self.fc43 = nn.Linear(68,68)
        self.fc44 = nn.Linear(68,68)
        self.fc45 = nn.Linear(68,68)
        #########
        self.fc51 = nn.Linear(68,68)
        self.fc52 = nn.Linear(68,68)
        self.fc53 = nn.Linear(68,68)
        self.fc54 = nn.Linear(68,68)
        self.fc55 = nn.Linear(68,68)
        #########
        self.fc6  = nn.Linear(68*68, 68*68)
        #########
        self.fcy = nn.Linear(latent_dim,1)
    def encode(self, x):
        h11 = F.relu(self.fc11(x))
        h12 = F.relu(self.fc12(x))
        h11 = self.drop_layer(h11)
        return self.fc21(h11), self.fc22(h12)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h31 = self.fc31(z)
        h32 = torch.sigmoid(self.fc32(z))
        h33 = torch.sigmoid(self.fc33(z))
        h34 = torch.sigmoid(self.fc34(z))   
        h35 = torch.sigmoid(self.fc35(z))
        ########
        h41 = torch.sigmoid(self.fc41(h31))
        h42 = torch.sigmoid(self.fc42(h32))
        h43 = torch.sigmoid(self.fc43(h33))
        h44 = torch.sigmoid(self.fc44(h34))
        h45 = torch.sigmoid(self.fc45(h35))
        ########
        h51 = (self.fc51(h41))
        h52 = (self.fc52(h42))
        h53 = (self.fc53(h43))
        h54 = (self.fc53(h44))
        h55 = (self.fc53(h45))
        ########
        h51 = torch.bmm(h51.unsqueeze(2), h51.unsqueeze(1))
        h52 = torch.bmm(h52.unsqueeze(2), h52.unsqueeze(1))
        h53 = torch.bmm(h53.unsqueeze(2), h53.unsqueeze(1)) 
        h54 = torch.bmm(h54.unsqueeze(2), h54.unsqueeze(1)) 
        h55 = torch.bmm(h55.unsqueeze(2), h55.unsqueeze(1)) 
        ########
        h6 = h51 
        # h6 = torch.sigmoid(h6.view(-1,68*68))
        h6 = torch.sigmoid(self.fc6(h6.view(-1,68*68)))
        return h6.view(-1, 68*68)
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 68*68))
        z = self.reparameterize(mu, logvar)
        y = self.fcy(mu)
        return self.decode(z), mu, logvar, y



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, yhat, y):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 68*68), reduction='sum')
    NCE = F.mse_loss(yhat, y, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, latent_dim14
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.1 * (BCE + KLD) + NCE


def train(epoch, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, y,_) in enumerate(train_loader):
        data = data.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, yhat = model(data)
        fc11_params = torch.cat([x.view(-1) for x in model.fc11.parameters()])
        fc21_params = torch.cat([x.view(-1) for x in model.fc21.parameters()])
        fc11_l1_regularization = 0.01 * torch.norm(fc11_params, 1)
        fc21_l1_regularization = 0.01 * torch.norm(fc21_params, 1)
        loss = loss_function(recon_batch, data, mu, logvar, yhat, y.view(-1,1)) + fc11_l1_regularization + fc21_l1_regularization
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/len(train_loader.dataset)))


def test(epoch, train_loader):
    model.eval()
    test_loss = 0
    yhat_vec = []
    y_vec = []
    with torch.no_grad():
        for i, (data, y,_) in enumerate(train_loader):
            data = data.to(device)
            y = y.to(device)
            recon_batch, mu, logvar, yhat = model(data)
            test_loss += F.mse_loss(yhat,y.view(-1,1), reduction='sum').item()
            yhat_vec.append(yhat)
            y_vec.append(y)
        print('---> Test Error {}'.format(test_loss/len(torch.cat(yhat_vec))))
    return test_loss


device = torch.device("cuda")
model = VAE().to(device)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)


nepoch=500
for epoch in range(nepoch):
    train(epoch, train_loader)
    test(epoch, train_loader)

beta = model.fcy.weight.data.cpu().numpy().transpose()
bias = model.fcy.bias.data.cpu().numpy()
sigma = 1
scale = 2

yvec = np.linspace(scale*np.min(np.asarray(tensor_response)), scale*np.max(np.asarray(tensor_response)), num=10)
yvec = (yvec- bias)/ (sigma**2)
nrep = 1

zlist = []
for i in range(len(yvec)):
    y = yvec[i]
    muzy = y * (np.matmul(np.linalg.inv(np.eye(beta.shape[1]) +  np.matmul(beta,beta.transpose())/(sigma**2)), beta)).flatten()
    print(muzy)
    sigmazy = np.linalg.inv(np.eye(beta.shape[0]) +  np.matmul(beta,beta.transpose())/(sigma**2))
    zmat = np.random.multivariate_normal(muzy, sigmazy, (nrep, 1))
    zlist.append(zmat)


for i in range(len(zlist)):
    zmat = zlist[i]
    zmat = zmat.squeeze()
    zmat = torch.from_numpy(np.float32(zmat)).to(device)
    with torch.no_grad():
        model.eval()
        recon = model.decode(zmat.reshape(-1,latent_dim))
        recon = np.mean(recon.cpu().detach().numpy(),0)
        plt.imshow(recon.reshape(68,68))
        plt.savefig('results/rec-{}.png'.format(i))


