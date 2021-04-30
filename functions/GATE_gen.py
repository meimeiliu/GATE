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
        super(VAE, self).__init__()
        self.fc11 = nn.Linear(68*68,400)
        self.fc21 = nn.Linear(400,latent_dim)
        self.fc22 = nn.Linear(400,latent_dim)
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
        h1 = F.relu(self.fc11(x))
        h1 = self.drop_layer(h1)
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h31 = torch.sigmoid(self.fc31(z))
        h41 = torch.sigmoid(self.fc41(h31))
        h32 = torch.sigmoid(self.fc32(z))
        h42 = torch.sigmoid(self.fc42(h32))
        h33 = torch.sigmoid(self.fc33(z))
        h43 = torch.sigmoid(self.fc43(h33))
        h34 = torch.sigmoid(self.fc34(z))
        h44 = torch.sigmoid(self.fc44(h34))
        h35 = torch.sigmoid(self.fc35(z))
        h45 = torch.sigmoid(self.fc45(h35))
        ########
        h51 = torch.sigmoid(self.fc51(h41))
        h52 = torch.sigmoid(self.fc52(h42))
        h53 = torch.sigmoid(self.fc53(h43))
        h54 = torch.sigmoid(self.fc53(h44))
        h55 = torch.sigmoid(self.fc53(h45))
        ########
        h51 = torch.bmm(h51.unsqueeze(2), h51.unsqueeze(1))
        h52 = torch.bmm(h52.unsqueeze(2), h52.unsqueeze(1))
        h53 = torch.bmm(h53.unsqueeze(2), h53.unsqueeze(1)) 
        h54 = torch.bmm(h54.unsqueeze(2), h54.unsqueeze(1)) 
        h55 = torch.bmm(h55.unsqueeze(2), h55.unsqueeze(1)) 
        ########
        h5 = F.relu(h51 + h52 + h53 + h54 + h55)
        h6 = torch.sigmoid(self.fc6(h5.view(-1,68*68)))
        return h6.view(-1, 68*68)
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 68*68))
        z = self.reparameterize(mu, logvar)
        y = self.fcy(mu)
        return self.decode(z), mu, logvar, y



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, yhat, y):
    if loss_type=="binary":
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 68*68), reduction='sum')
    if loss_type=="poisson":
        BCE = F.poisson_nll_loss(recon_x , x, reduction='sum', log_input=True)
    NCE = F.mse_loss(yhat, y, reduction='sum')
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return  (BCE + KLD)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y,label) in enumerate(train_loader):
        data = data.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, yhat = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, yhat, y.view(-1,1))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss/len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, y,label) in enumerate(train_loader):
            data = data.to(device)
            y = y.to(device)
            recon_batch, mu, logvar, yhat = model(data)
            test_loss += F.mse_loss(yhat,y.view(-1,1),reduction='sum').item()
        print('---> Test Error {}'.format(test_loss/len(train_loader.dataset)))



device = torch.device("cuda")
model = VAE().to(device)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

nepoch=2000
for epoch in range(nepoch):
    train(epoch)


with torch.no_grad():
    sample = torch.randn(10, 68).to(device)
    sample = model.decode(sample).cpu()
    for i in range(len(sample)):
        plt.imshow(sample[i].reshape(68,68))
        plt.savefig('results/{}.png'.format(i))


num_elements = len(train_loader.dataset)
num_batches = len(train_loader)
batch_size = train_loader.batch_size
mu_out = torch.zeros(num_elements, latent_dim)
label_out =  torch.zeros(num_elements)

with torch.no_grad():
    model.eval()
    for i, (data,trait,label) in enumerate(train_loader):
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        data = data.to(device)
        recon_batch, mu, logvar, yhat = model(data)
        mu_out[start:end] = mu
        label_out[start:end] = label


X = mu_out.numpy()
pca = PCA(n_components=3)
pca.fit(X.transpose())
pca_results = pca.components_.transpose()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(pca_results[ :,0], pca_results[:, 1], c =label_out.numpy().astype(int),alpha=0.8);
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
plt.savefig('results/pca.png')
plt.close()


