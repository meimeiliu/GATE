from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from scipy.io import loadmat
from torch.autograd import Variable
import numpy as np
import torch.utils.data as utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt


device = torch.device("cuda")


torch.manual_seed(11111)
###Load data set
device = torch.device("cuda")
dat_mat = loadmat('./data/HCP_samples.mat')
tensor = dat_mat['loaded_tensor_sub']

###Load the adjacentcy matrix
A_mat = np.mean(np.squeeze(tensor[18:86, 18:86,1,:]), axis=2)
A_mat = A_mat + A_mat.transpose()



### Choose the proper type of loss function 
loss_type = "poisson"
### If the loss_type = "poisson", set the proper offset for the mean
offset = 100 
### Set the neighborhood size for GATE
n_size = 32

### Load networks
net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:,:,0,i] + np.transpose(tensor[:,:,0,i]))
    np.fill_diagonal(ith,np.mean(ith, 0))
    ith = ith[18:86, 18:86]
    ith = ith.flatten()
    ith = ith/offset
    net_data.append(ith)


###Load the phenotype 
trait_data = np.squeeze(torch.load('./data/trait.pt')[:,0])
trait_data = trait_data.numpy()
trait_data = (trait_data-np.mean(trait_data))/np.std(trait_data)
trait_data = torch.from_numpy(trait_data)
trait_data = trait_data.float()
# 0 - Oral Reading Recognition Test
# 1 - Picture Vocabulary Test 
# 2 - Line Orientation: Total number correct
# 3 - Line Orientation: Total positions off for all trials 
batch_size = 5
tensor_y = torch.stack([torch.Tensor(i) for i in net_data])
y = utils.TensorDataset(tensor_y, trait_data) # create your datset
train_loader = utils.DataLoader(y, batch_size) 


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)



class GraphCNN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphCNN, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.mask_flag = True 
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 68
        self.w = Variable(torch.randn(5, 1), requires_grad=True).cuda()
        self.a = Variable(torch.randn(68*68), requires_grad=True).cuda()
        self.fc1 = nn.Linear(68*68, 256)
        self.fc21 = nn.Linear(256, latent_dim)
        self.fc22 = nn.Linear(256, latent_dim)
        self.fc31 =  nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)
        self.fc41 = GraphCNN(68,68)
        self.fc42 = GraphCNN(68,68)
        self.fc43 = GraphCNN(68,68)
        self.fc44 = GraphCNN(68,68)
        self.fc45 = GraphCNN(68,68)
        self.fcy1 = nn.Linear(latent_dim,1)
        self.fcintercept = nn.Linear(68*68, 68*68)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h31= F.sigmoid(self.fc31(z))
        h31= F.sigmoid(self.fc41(h31))
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
        h32 = F.sigmoid(self.fc32(z))
        h32 = F.sigmoid(self.fc42(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))
        h33 = F.sigmoid(self.fc33(z))
        h33 = F.sigmoid(self.fc43(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = F.sigmoid(self.fc34(z))
        h34 = F.sigmoid(self.fc44(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = F.sigmoid(self.fc35(z))
        h35 = F.sigmoid(self.fc45(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        # h30 = h31_out + h32_out + h33_out + h34_out + h35_out
        h30 = torch.cat((h31_out.view(-1,68*68,1), h32_out.view(-1,68*68,1), h33_out.view(-1,68*68,1), h34_out.view(-1,68*68,1), h35_out.view(-1,68*68,1)), 2)
        h30 = torch.bmm(h30, self.w.expand(batch_size,5,1))
        h30 = self.a.expand(batch_size, 68*68) + h30.view(-1,68*68)
        # h30 = (h30)
        h30 = self.fcintercept(h30)
        return h30.view(-1, 68*68)
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 68*68))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        trait = self.fcy1(mu)
        return recon.view(-1, 68*68), mu, logvar, trait.view(-1,1)
    def set_mask(self, masks):
        self.fc41.set_mask(masks[0])
        self.fc42.set_mask(masks[1])
        self.fc43.set_mask(masks[2])
        self.fc44.set_mask(masks[3])
        self.fc45.set_mask(masks[4])
        # self.fcintercept.set_mask(masks[5])



def loss_function(recon_x, that, x, t , mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x , reduction='sum')
    BCE = F.poisson_nll_loss(recon_x , x, reduction='sum', log_input=True)
    NCE = F.mse_loss(that, t, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.01*(BCE + KLD) + NCE


def train(epoch, train_loader_k):
    model.train()
    train_loss = 0
    for batch_idx, (data, trait) in enumerate(train_loader_k):
        data = data.to(device)
        trait = trait.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, traithat = model(data.view(-1,68*68))
        loss = loss_function(recon_batch.view(-1,68*68), traithat.view(-1,1), data.view(-1,68*68), trait.view(-1,1), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader_k.dataset)))


def test(epoch, test_loader_k, test_index):
    model.eval()
    test_loss = 0
    pred_val = torch.zeros(len(test_index))
    test_val = torch.zeros(len(test_index))
    with torch.no_grad():
        for i, (data, trait) in enumerate(test_loader_k):
            data = data.to(device)
            trait = trait.to(device)
            recon_batch, mu, logvar, traithat = model(data)
            test_loss += F.mse_loss(traithat, trait.view(-1,1) ,reduction='sum')
            start = i*batch_size
            end = start + batch_size
            if i == len(test_loader_k) - 1:
                end = len(test_index)
            pred_val[start:end] = traithat.view(-1).cpu()
            test_val[start:end] = trait.view(-1).cpu()
    test_loss /= len(test_index)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, pred_val, test_val



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
masks = []
mask_2NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_2NN)).float())
mask_4NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_4NN)).float())
mask_8NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_8NN)).float())
mask_16NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_16NN)).float())
mask_16NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_16NN)).float())
mask_intercept = np.identity(68*68)
masks.append(torch.from_numpy(np.float32(mask_intercept)).float())
model.set_mask(masks)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 5
learning_rate=0.01


nepoch = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for j in range(nepoch):
    train(j, train_loader)
    valid_loss, traithat, trait = test(1, train_loader, np.arange(len(train_loader.dataset)))
    print(valid_loss)


###########################################################
### Generate brain networks
###########################################################
beta = model.fcy1.weight.data.cpu().numpy().transpose()
bias = model.fcy1.bias.data.cpu().numpy()
sigma = np.std(trait_data.numpy())
scale = 5

yvec = np.linspace(scale*min(trait_data.numpy()), scale*max(trait_data.numpy()), num=10)
yvec = (yvec- bias)/ (sigma**2)
nrep = 1

zlist = []
for i in range(len(yvec)):
    y = yvec[i]
    muzy = y * (np.matmul(np.linalg.inv(np.eye(len(beta)) +  np.matmul(beta,beta.transpose())/(sigma**2)), beta)).flatten()
    print(muzy)
    sigmazy = np.linalg.inv(np.eye(len(beta)) +  np.matmul(beta,beta.transpose())/(sigma**2))
    zmat = np.random.multivariate_normal(muzy, sigmazy, (nrep, 1))
    # zmat = np.random.multivariate_normal(muzy, sigmazy, (nrep, 1))
    # zlist.append(zmat)
    zlist.append(muzy)



recon_list = []
plt.clf()
for i in range(len(zlist)):
    zmat = zlist[i] 
    zmat = zmat.squeeze()
    print(np.matmul(zmat, beta))
    zmat = np.repeat(zmat.reshape(1,68),5, axis=0)
    # print(np.matmul(muzy, beta))
    zmat = torch.from_numpy(np.float32(zmat)).to(device)
    with torch.no_grad():
        model.eval()
        recon = model.decode(zmat.reshape(-1,68))
        recon = np.mean(recon.cpu().detach().numpy(),0)
        recon_list.append(recon)
        plt.imshow(recon.reshape(68,68)>1)
        plt.savefig('results/rec-{}.png'.format(i))

