# GATE and reGATE 

We develop a novel nonlinear latent factor model to characterize the population distribution of brain connectomes across individuals and depending on human traits. GATE outputs two layers of low dimensional nonlinear latent representations: one on the individuals that can be used as a summary score for visualization and prediction of human traits of interest; and one on the nodes for characterizing the network structure of each individual based on a latent space model. A supervised model reGATE is proposed to analyze the relationship between human traits and brain connectomes. 
GATE/reGATE is developed based on a deep neural network framework and implemented via a stochastic variational Bayesian algorithm. It is computationally efficient, and can be easily extended to massive networks with very large number of nodes (brain ROIs). 


## Installation
Our GATE and reGATE pipeline has several dependencies including:

1. pytorch-gpu (require cuda library)
2. numpy/scikit-learn/pandas/matplotlib
3. networkx

We recommend using Anaconda, which has many of these installed, and allows ready installation of the remaining packages through conda or pip. This project is written in python3. 

## Creating an virtual environment using Anaconda

1. Create a virtual conda environment: conda create -n gate python=3.7 anaconda
2. Activate that environment: source activate gate
3. Install the libraries into the gate environment: pip install -r requirements.txt


## Usage 
Our workflow takes as input a series of 2D network performs the following:

1. unsupervised GATE: GATE.py
2. a general version of unsupervised GATE if you do not have a geometric distance matrix for the network: GATE_gen.py
3. regression GATE for prediction: reGATE_pred.py
4. a general version of reGATE for prediction if you do not have a geometric distance matrix for the network: reGATE_pred_gen.py
5. regression GATE for reconstruction: reGATE_rec.py
6. a general version of regression GATE for reconstruction without the geometric distance matrix for the network: reGATE_rec_gen.py

In addition, we provide a series of tools for analyzing the results:
1. extract the latent space for visualization: extract.py
2. generate random network: generate_random_graph.py

## Data set: 

1. HCP_samples.mat includes 100 randomly choosen samples from HCP data set
loaded_tensor_sub(:,:,1,:) are network data
loaded_tensor_sub(:,:,2,:) are adjacentcy matrix

2. trait.pt includes  1 - Oral Reading Recognition Test, 2 - Picture Vocabulary Test, 3 - Line Orientation: Total number correct, and 4 - Line Orientation: Total positions off for the 100 samples.


## Examples
```
from __future__ import print_function
import torch
import torch.utils.data as utils
import os  

#################################################################
###A general example without adjacency matrix
#################################################################
##Generate random networks
os.system('python3 tools/generate_random_graph.py')

###Create directory to store the results
results_path = "./results"
try:
    os.mkdir(results_path)
except OSError:
    print ("Creation of the directory %s failed" % results_path)
else:
    print ("Successfully created the directory %s " % results_path)

###Run unsuperived GATE with GATE_gen.py output pca plot and some generated networks
os.system('python3 functions/GATE_gen.py')

###Run superived GATE with reGATE_gen.py for prediction
os.system('python3 functions/reGATE_pred_gen.py')

###Run superived GATE with reGATE_gen.py for reconstruct network conditioned on trait
os.system('python3 functions/reGATE_rec_gen.py')

#################################################################
###real example with adjacency matrix
#################################################################

##Generate sampled HCP data network with 
os.system('python3 tools/generate_random_graph.py')

###Create directory to store the results
results_path = "./results"
try:
    os.mkdir(results_path)
except OSError:
    print ("Creation of the directory %s failed" % results_path)
else:
    print ("Successfully created the directory %s " % results_path)

###Run unsuperived GATE with GATE.py output pca plot and some generated networks
os.system('python3 functions/GATE.py')

###Run superived GATE with reGATE_gen.py for prediction
os.system('python3 functions/reGATE_pred.py')

###Run superived GATE with reGATE_gen.py for reconstruct network conditioned on trait
os.system('python3 functions/reGATE_rec.py')
```