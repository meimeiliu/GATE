from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time
t_start = time.time()

### Choose the proper type of loss function 
loss_type = "binary"



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
        self.fcy1 = nn.Linear(latent_dim,1024)
        self.fcy2 = nn.Linear(1024,1)
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
        y = F.sigmoid(self.fcy1(mu))
        y = self.fcy2(y)
        return self.decode(z), mu, logvar, y



# Reconstruction + KL divergence + regression losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, yhat, y):
    if loss_type=="binary":
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 68*68), reduction='sum')
    if loss_type=="poisson"
        BCE = F.poisson_nll_loss(recon_x , x, reduction='sum', log_input=True)
    NCE = F.mse_loss(yhat, y, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.01 * (BCE + KLD) + NCE


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
    return test_loss, torch.cat(yhat_vec), torch.cat(y_vec)


device = torch.device("cuda")
model = VAE().to(device)
lr = 0.001
optimizer = optim.Adam([{'params': weight_p, 'weight_decay':1e-5},
                      {'params': bias_p, 'weight_decay':1e-5}], lr=lr)


nepoch=50

from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
kf = KFold(n_splits=5,shuffle=True, random_state=123456)
cv_loss = np.zeros([5,nepoch])
pred_val = np.zeros(len(train_loader.dataset))
test_val = np.zeros(len(train_loader.dataset))


for i, (train_index, test_index) in enumerate(kf.split(response_std)):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_train = len(train_loader.dataset)
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(test_index)
    train_loader_k = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader_k = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, sampler=valid_sampler)
    for j in range(nepoch):
        train(j, train_loader_k)
        valid_loss, traithat, trait = test(1, valid_loader_k)
        cv_loss[i,j] = valid_loss/len(test_index)
        print('Testing Error:{}'.format( valid_loss/len(test_index)))
    pred_val[test_index] = traithat.cpu().view(80)
    test_val[test_index] = trait.cpu().view(80)



print("--- %s seconds ---" % (time.time() - t_start))


np.mean(np.amin(cv_loss,1))


