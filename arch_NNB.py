# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# This module provides an illustration of labeling FE instances with KNN
# and an imbalanced CIFAR-10 training set.  The source file is
# created using Extract_FE.py.

################################################################
#settings

#number of nearest neighbors
num_neigh = 5

#file location with source data
source_file = ".../CE_cifar10_train.csv"

save_file = ".../CE_cif_trn_NNB.csv"

################################################################

pdf = pd.read_csv(source_file)
x = pdf.to_numpy()

#extract FE (features)
xtrn = x[:,3:]

#extract target labels
xtrnlab = x[:,0]

#extract CNN predictions
xtrn_pred = x[:,1]


xtrn_idx = np.arange(len(x)) 

nn = NearestNeighbors(n_neighbors=num_neigh+1)
nn.fit(xtrn)
dist, ind = nn.kneighbors(xtrn)

# nearest neighbors of each training instance
neigh = np.empty([x.shape[0],num_neigh],dtype=np.int8)
neigh_ind = []


categories = np.empty([x.shape[0]],dtype=np.int8)
categories.shape

#categories or archetypes
#0 = safe - 4-5 same class
#1 = border 2-3 same class
#2 = rare 1 same class
#3 = noise 0 same class

for i in range(len(ind)):
            
            lab = xtrnlab[i]
            
            y = xtrnlab[ind[i][1:]]
            
            z = sum(y == lab)
            
            if z >= 4:
                #safe
                categories[i]=0
                
            elif z == 3 or z==2:
                #border
                categories[i]=1
            
            elif z == 1:
                #rare or sub-concept
                categories[i]=2    
            
            else:
                #outlier
                categories[i]=3
                
            neigh[i,:]=y
            
            neigh_ind.append(xtrn_idx[ind[i][1:]])

#prepare data for saving

#index locations
pd_idx = pd.DataFrame(data=xtrn_idx,columns=['idx'])
#target labels
pd_lab = pd.DataFrame(data=xtrnlab,columns=['actual'])
#prediction lables
pd_pred = pd.DataFrame(data=xtrn_pred,columns=['pred'])
#archetypes
pd_cat = pd.DataFrame(data=categories,columns=['cats'])

#headings for neigbor labels
Ns = ['N' + str(i) for i in range(num_neigh)]
#headings for neighbor indices
Is = ['I' + str(i) for i in range(num_neigh)]

pd_neigh = pd.DataFrame(data=neigh,columns=Ns)
pd_idx1 = pd.DataFrame(data=neigh_ind,columns=Is)

mem_bank = pd.concat([pd_lab,pd_idx, pd_pred,pd_cat, 
        pd_neigh,pd_idx1],axis=1)

mem_bank.to_csv(save_file,index=False)

##########################################################################
