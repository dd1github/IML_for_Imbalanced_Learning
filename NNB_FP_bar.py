# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



#inputs
#the inputs are based on CIFAR-10 as an illustration
###################################################################

#number of classes
num_cls = 10

#class names
classes = ('Plane', 'Car', 'Bird', 'Cat',
           'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')


#file location with source data
#this file is created with arch_NNB.py
source_file = ".../CE_cif_trn_NNB.csv"

save_file = ".../NNB.pdf"

#####################################################
#train nnbs

pdf = pd.read_csv(source_file)
x = pdf.to_numpy()

#extract targets
tar = x[:,0]

#extract CNN predictions
pred = x[:,2] 

#extract neighbors
nnb1 = x[:,4:9] 

#track false positives
incorrect = np.empty([num_cls,num_cls],dtype=float)


for i in range(num_cls):
    
    tars = tar[tar==i]
    preds = pred[tar==i]
    nnb1s = nnb1[tar==i]
    
    tarc = tars[preds!=i]
    preds1 = preds[preds!=i]
    nnb1s = nnb1s[preds!=i]
    
    nnb1s = nnb1s.flatten()
    
    predlist = np.unique(nnb1s)
    
    for n in range(num_cls):
        if n in predlist:
            
            num = len(nnb1s[nnb1s==n])
            
        else:
            num = 0
        
        incorrect[i,n]=num   
        

for i in range(num_cls):
    incorrect[i,i]=0

in_sum = np.sum(incorrect, axis=1)
in_sum = np.where(in_sum==0,.0000001,in_sum)

inc_per = incorrect/in_sum.reshape(-1,1)

classes1 = np.arange(num_cls)

df1 = pd.DataFrame(data=classes1,columns=['class'])

cls_title = [i for i in range(num_cls)]


df2 = pd.DataFrame(data=inc_per, columns=cls_title)

df3 = pd.concat([df1,df2],axis=1)

sns.set(style='white')

df3.set_index('class').plot(kind='bar', stacked=True,
    color=['steelblue', 'red','green', 'cyan',
    'yellow', 'tan', 'orange', 'dodgerblue','violet', 'silver'])

plt.title('CIFAR-10: Nearest Neighbors (Training)', fontsize=14)

plt.legend(classes, loc='center left', bbox_to_anchor=(1, 0.5))

labs=classes

x1 = np.arange(num_cls)

plt.xticks(x1,labs)

plt.xlabel('Classes')
plt.ylabel('Adversary Class Neighbor Percent')

plt.xticks(rotation=0)

plt.savefig(save_file,bbox_inches='tight')

plt.show()

