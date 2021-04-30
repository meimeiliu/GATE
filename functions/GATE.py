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

batch_size = 5
tensor_y = torch.stack([torch.Tensor(i) for i in net_data])
y = utils.TensorDataset(tensor_y) # create your datset
train_loader = utils.DataLoader(net_data, batch_size) 




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
        latent_dim = 8
        self.fc11 =  nn.Linear(68*68, 1024)
        self.fc12 =  nn.Linear(68*68, 1024)
        self.fc111 = nn.Linear(1024,128)
        self.fc222 = nn.Linear(1024,128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)
        self.fc4 = GraphCNN(68,68)
        self.fc5 = GraphCNN(68,68)
        self.fc6 = GraphCNN(68,68)
        self.fc7 = GraphCNN(68,68)
        self.fc8 = GraphCNN(68,68)
        self.fcintercept = GraphCNN(68*68, 68*68)
    def encode(self, x):
        h11 = F.relu(self.fc11(x))
        h11 = F.relu(self.fc111(h11))
        h12 = F.relu(self.fc12(x))
        h12 = F.relu(self.fc222(h12))
        return self.fc21(h11), self.fc22(h12)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h31= F.sigmoid(self.fc3(z))
        h31= F.sigmoid(self.fc4(h31))
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
        h32 = F.sigmoid(self.fc32(z))
        h32 = F.sigmoid(self.fc5(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))
        h33 = F.sigmoid(self.fc33(z))
        h33 = F.sigmoid(self.fc6(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = F.sigmoid(self.fc34(z))
        h34 = F.sigmoid(self.fc7(h33))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = F.sigmoid(self.fc35(z))
        h35 = F.sigmoid(self.fc8(h33))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        h30 = F.sigmoid(h31_out + h32_out + h33_out + h34_out)
        h30 = h30.view(-1, 68*68)
        h30 = self.fcintercept(h30)
        return h30.view(-1, 68*68), h31+h32+h33+h34
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 68*68))
        z = self.reparameterize(mu, logvar)
        recon, x_latent = self.decode(z)
        return recon.view(-1, 68*68), mu, logvar, x_latent
    def set_mask(self, masks):
        self.fc4.set_mask(masks[0])
        self.fc5.set_mask(masks[1])
        self.fc6.set_mask(masks[2])
        self.fc7.set_mask(masks[3])
        self.fc8.set_mask(masks[4])
        self.fcintercept.set_mask(masks[5])



def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x , reduction='sum')
    BCE = F.poisson_nll_loss(recon_x , x, reduction='sum', log_input=True)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(train_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))




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


for epoch in range(100):
    # if epoch < 5:
    #     model.fc11.requires_grad = False
    #     model.fc21.requires_grad = False
    # if epoch >= 5:
    #     model.fc11.requires_grad = True
    #     model.fc21.requires_grad = True
    train(epoch)


with torch.no_grad():
    sample = torch.randn(10, 8).to(device)
    (sample,_) = model.decode(sample)
    sample = sample.cpu()
    for i in range(len(sample)):
        plt.imshow(sample[i].reshape(68,68))
        plt.savefig('results/{}.png'.format(i))

torch.cuda.empty_cache()

