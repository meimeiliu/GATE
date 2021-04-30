import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import manifold, datasets
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib 
# configure backend here
matplotlib.use('Agg')



latent_dim=68
num_elements = len(train_loader.dataset)
num_batches = len(train_loader)
batch_size = train_loader.batch_size
mu_out = torch.zeros(num_elements, latent_dim)
logvar_out = torch.zeros(num_elements,latent_dim)
recon_out = torch.zeros(num_elements,68*68)
trait_predict = torch.zeros(num_elements, 1)


###########################################################
### Extract the latent features
###########################################################
with torch.no_grad():
    model.eval()
    for i, (data,trait) in enumerate(train_loader):
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        data = data.to(device)
        trait = trait.to(device)
        recon_batch, mu, logvar, traithat = model(data)
        mu_out[start:end] = mu
        logvar_out[start:end] = logvar
        # x_latent_out[start:end] = x_latent
        recon_out[start:end] =recon_batch
        trait_predict[start:end] = traithat


###Save to matlab file
np.save('mu-{}.npy'.format(feature_id), mu_out.detach().numpy())

###Save to matlab file
import scipy.io
scipy.io.savemat('mu-{}-un.mat'.format(feature_id), mdict={'mu': mu_out.detach().numpy()})
