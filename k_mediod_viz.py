# -*- coding: utf-8 -*-

from sklearn_extra.cluster import KMedoids
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from imbalance_cifar import IMBALANCECIFAR10


#inputs
#the inputs are based on CIFAR-10 as an illustration
###################################################################

#number of classes
num_class = 10 

#number of archetypes
num_arch = 4

#file location with source data
#this file is created with arch_NNB.py
source_file = ".../CE_cif_trn_NNB.csv"

#source file with instance embeddings (FE)
#file created with Extract_FE.py
source_FE = ".../CE_cifar10_train.csv"

#folder to save the prototypical images
save_folder = ".../protos/"

# class target index labels
select = np.array([0,9]) 

#class labels of interest
classes = ['Plane','Truck']

#number of classes selected
len_cls_select = len(classes)

#archetype labels
arch = ['Safe','Border','Rare','Noise']

#location where image dataset stored
data_root = '.../data/'

#######################################################################

train_dataset = IMBALANCECIFAR10(root=data_root, 
        imb_type='exp', imb_factor=0.01,
        rand_number=0, train=True, download=True)


pd_ce = pd.read_csv(source_file)
x = pd_ce.to_numpy()

#extract targets
tar = x[:,0]
#extract indices
idx = x[:,1]
#extract CNN class predictions
pred = x[:,2]
#extract categories or archetypes
cat = x[:,3]


pdf = pd.read_csv(source_FE)
x1 = pdf.to_numpy()

#extract FE
xfeat = x1[:,3:]

medoids = np.empty([len_cls_select,num_arch],dtype=int)

count = 0
for i in select:
    tars = tar[tar==i]
    idxs = idx[tar==i]
    cats = cat[tar==i]
    feats = xfeat[tar==i]
    
    for c in range(num_arch):
        ctars = tars[cats==c]
        cfeats = feats[cats==c]
        cidx = idxs[cats==c]
        kmedoids = KMedoids(n_clusters=1, random_state=0).fit(cfeats)
        med = kmedoids.medoid_indices_
        medoids[count,c]=cidx[med]
    
    count+=1


for i in range(len(select)):
    for j in range(num_arch):
        
        img_idx = medoids[i,j]
        
        img_idx = img_idx.astype(int)
        
        plt.imshow(train_dataset.data[img_idx])
        plt.title(classes[i])
       
        plt.axis('off')
        
        f3 = save_folder + classes[i] +'_' + arch[j] + '.pdf'
        
        plt.savefig(f3,bbox_inches='tight')
        plt.show()








