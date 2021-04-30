#importing the networkx library 
import networkx as nx 
from scipy.sparse import csr_matrix
import random
import numpy as np
#importing the matplotlib library for plotting the graph 
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
import torch.utils.data as utils
import torch
from scipy.io import loadmat
from numpy import linalg as LA
import scipy.io

###########################################################
## Set the number of nodes in 
###########################################################
nvex = 68
random.seed(123456)

net_data = []
response = []
nrep = 100
A_erdos = np.zeros([nvex,nvex])
A_small = np.zeros([nvex,nvex])
A_commu = np.zeros([nvex,nvex])
A_scale = np.zeros([nvex,nvex])
for i in range(nrep):
    ##Erdos network
    G = nx.gnm_random_graph(nvex,400, seed=123) 
    A = nx.adjacency_matrix(G)
    A = csr_matrix.todense(A)
    A_erdos = A_erdos + A
    net_data.append(A.reshape(nvex,nvex))
    ##Small world network
    G = nx.watts_strogatz_graph(nvex, 10, 0.5)
    A = nx.adjacency_matrix(G)
    A = csr_matrix.todense(A)
    A_small = A_small + A
    net_data.append(A.reshape(nvex,nvex))
    ##Random Community network
    G = nx.random_partition_graph([34, 34], .25,.01)
    A = nx.adjacency_matrix(G)
    A = csr_matrix.todense(A)
    A_commu = A_commu + A
    net_data.append(A.reshape(nvex, nvex))
    ##Scale free network
    G = nx.barabasi_albert_graph(nvex, 5)
    A = nx.adjacency_matrix(G)
    A = csr_matrix.todense(A)
    A_scale = A_scale + A
    net_data.append(A.reshape(nvex,nvex))



alpha = np.zeros(68)
alpha[0:17]=1


net_data = []
response = []
label = []
nrep = 100
for i in range(nrep):
    A = np.random.binomial(1,0.8*A_erdos/nrep, A.shape)
    # A = np.matmul(A,A)
    net_data.append(A.reshape(nvex,nvex))
    response.append(np.asscalar(np.matmul(np.matmul(alpha.reshape(1,68), A), alpha)))
    label.append(0.0)
    A = np.random.binomial(1,A_small/nrep, A.shape)
    # A = np.matmul(A,A)
    net_data.append(A.reshape(nvex,nvex))
    response.append(np.asscalar(np.matmul(np.matmul(alpha.reshape(1,68), A), alpha)))
    label.append(1.0)
    A = np.random.binomial(1,A_commu/nrep, A.shape)
    # A = np.matmul(A,A)
    net_data.append(A.reshape(nvex,nvex))
    response.append(np.asscalar(np.matmul(np.matmul(alpha.reshape(1,68), A), alpha)))
    label.append(2.0)
    A = np.random.binomial(1,A_scale/nrep, A.shape)
    # A = np.matmul(A,A)
    response.append(np.asscalar(np.matmul(np.matmul(alpha.reshape(1,68), A), alpha)))
    net_data.append(A.reshape(nvex,nvex))
    label.append(3.0)


response_std = (np.asarray(response) -  np.mean(np.asarray(response)))/np.std(np.asarray(response))
net_mean = np.mean(net_data, 0)
tensor_net = torch.stack([torch.Tensor((i-net_mean).reshape(68*68)) for i in net_data])
# tensor_net = tensor_net.to(dtype=torch.int32)
tensor_response =torch.from_numpy(np.float32(response_std))

torch.save([tensor_net,tensor_response,torch.from_numpy(np.asarray(label))], 'data.pt')

###########################################################
##Output to matlab
###########################################################
# scipy.io.savemat('sim_data.mat', mdict = {'X': mat_out, "Y": response_std})
# y1 = np.asarray(response_std)[np.arange(0,400,4)]
# y2 = np.asarray(response_std)[np.arange(1,400,4)]
# y3 = np.asarray(response_std)[np.arange(2,400,4)]
# y4 = np.asarray(response_std)[np.arange(3,400,4)]


