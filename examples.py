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
