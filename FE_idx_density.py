# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This module provides an illustration of visualizing FE overlapping 
# indices and densities with an imbalanced CIFAR-10 training set. 
# The source file is created using Extract_FE.py.

################################################################
#settings

num_cls = 10
num_feats = 64
topk=10

#file location with source data
source_file = ".../CE_cifar10_train.csv"

#location to save FE index visualization
save_file_idx = ".../FE_idx.pdf"

#location to save FE density visualization
save_file_density = ".../FE_density.pdf"

classes = ('Plane', 'Car', 'Bird', 'Cat',
           'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# colors for FE top-10 
colors =['lightgray','darksalmon','yellow','tan','lime','wheat',
        'cyan','lavender','lightblue','pink']

#select a reference class for purposes of visualizing class densities
#here, trucks = 9
ref_cls = 9

################################################################
#FE index overlap

pdf = pd.read_csv(source_file)
x = pdf.to_numpy()

#extract FE
feats = x[:,3:]

#extract target labels
labs = x[:,0]

mu_all = np.empty([num_cls,num_feats],dtype=float)

#calculate FE means for each FE in each class
for i in range(num_cls):

    data = feats[labs==i]
    
    mu = np.mean(data,axis=0)
    
    mu_all[i,:]=mu


#reverse sorted indices
rev_idxs = np.zeros((num_cls,topk),dtype=int)

rev_vals = np.zeros((num_cls,topk),dtype=float)
fe_counts = np.zeros(num_cls,dtype=int)

for i in range(num_cls):
    
    r_mu = mu_all[i,:]
    mu_sum = np.sum(r_mu)
    
    r_mu_sort = np.sort(r_mu)
    r_mu_argsort = np.argsort(r_mu)
    rev_sort = r_mu_sort[::-1]
    rev_argsort = r_mu_argsort[::-1]
    
    rev_idxs[i,:]=rev_argsort[:10]
    
    rev_sum = np.sum(rev_sort[:10])
    rev_sort_topk = rev_sort[:10] / rev_sum
    
    rev_vals[i,:]=rev_sort_topk
    



cols = np.arange(topk)

cols = [str(i) for i in cols]


df = pd.DataFrame(data=rev_vals,columns=cols)

df.index = classes

rT = rev_idxs.T

ax = df.plot(kind='bar', stacked=True, figsize=(8, 6), rot=0, 
             xlabel='Class', ylabel='Percent of Top-K',color=colors)

plt.title('Top-K=10 FE (Feature Maps)', fontsize=14)

ax.legend([],[], frameon=False)

count=0
for c in ax.containers:

    labels = rT[count,:] 
        
    ax.bar_label(c, labels=labels, label_type='center')
    count+=1

plt.savefig(save_file_idx,bbox_inches='tight')

plt.show()



####################################################################
#FE density

#extract FE
xtrn = x[:,3:]

#extract targets
xtrnlab = x[:,0]

#extract predictions
xtrn_pred = x[:,1]

#FE for reference class
ref_feats = xtrn[xtrnlab==ref_cls]

ref_topk = rev_idxs[ref_cls,:]

ref_sort = np.argsort(ref_feats,axis=1)
rfeats_argst = ref_sort[:,::-1]
rfeats_argst = rfeats_argst[:,:10]


ref_topk_count =  np.zeros(topk,dtype=float)

for i in range(len(ref_topk)):
    rtopk = ref_topk[i]
    
    cnt = np.where(rfeats_argst==rtopk,1,0) 
    
    cnt = cnt.flatten()
    count = np.sum(cnt)
    ref_topk_count[i]=count
    
#adversary top-K
adv_topk = np.zeros((num_cls,topk),dtype=float)


for i in range(num_cls): 
    if i !=ref_cls:
        adv_feats = xtrn[xtrnlab==i]
        
        afeats_sort = np.argsort(adv_feats,axis=1)
        afeats_argst = afeats_sort[:,::-1]
        afeats_argst = afeats_argst[:,:10]
        
        for j in range(len(ref_topk)):
            rtopk = ref_topk[j]
            
            cnt = np.where(afeats_argst==rtopk,1,0) 
            
            cnt = cnt.flatten()
            count = np.sum(cnt)
            adv_topk[i,j]=count
 
atopk = adv_topk / ref_topk_count

atopk1 = atopk[:9,:]

####################################################
#scaling for purposes of visualization

scale_sum = np.sum(atopk1, axis=1)

max_sum = np.max(scale_sum)

min_sum = np.min(scale_sum)

f = str(max_sum)
a = f[::1].find('.')

f = str(min_sum)
b=f[::1].find('.')

new_topk = np.copy(atopk1)    

factor_reduction = 2.0 

if a-b > 1:
    for i in range(len(atopk1)):
        summed = atopk1[i,:]
        f = str(summed)
        c=f[::1].find('.')
        if c == a:
            new_topk[i,:]= new_topk[i,:] / factor_reduction

scale_sum = np.sum(new_topk, axis=1)

atopk2 = np.zeros((num_cls-1,topk),dtype=float)

for i in range(num_cls-1):
    at = atopk1[i,:]
    ats = np.sum(at)
    att = at / ats
    atopk2[i,:]=att

#################

cols = np.arange(topk)
cols = [str(i) for i in cols]

df = pd.DataFrame(data=new_topk,columns=cols)


classes = np.asarray(classes)
classes = np.delete(classes, ref_cls)
classes = classes.tolist()
classes = tuple(classes)


df.index = classes

rT = new_topk.T

ax = df.plot(kind='bar', stacked=True, figsize=(8, 6), rot=0, 
             xlabel='Class', ylabel='Density Ratio',color=colors)

plt.title('Top-K=10 FE Density Ratio', fontsize=14)

#legened
leg = np.arange(10)

rtk = [str(i) for i in leg]

count=0
for c in ax.containers:

    lc=rT[count,:]
    
    labels = []
    
    for i in lc:
        labels.append("%2.1f" % (i))
    labels = np.array(labels)
    
    den = new_topk[:,count]
    
    for i in range(len(den)):
        
        #note this may need to be adjusted based on visualization scaling
        if den[i] < 4.0: #factor_reduction 
            labels[i]=''
    
    ax.bar_label(c, labels=labels, label_type='center')
    
    h, labls = ax.get_legend_handles_labels()
    ax.legend(h,ref_topk)
    count+=1

plt.savefig(save_file_density,bbox_inches='tight')

plt.show()



















