# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set(style='white')

#inputs
#the inputs are based on CIFAR-10 as an illustration
###################################################################

#Location of class accuracies to plot
#this file is an output of Extract_FE.py
#here, we use a CIFAR-10 validation file which can be downloaded from our
#git site using a link
accur_file=".../cif10_val_cls_acc.csv"

#number of classes in dataset
num_cls = 10

#number of instances per class in dataset
num_instances_per_cls = np.array([5000, 2997, 1796, 1077, 645, 387,
            232, 139, 83, 50])

#path to save bar chart output
save_file = ".../accuracy.pdf"


##################################################################
pdf = pd.read_csv(accur_file)

x = np.arange(num_cls) 

sns.barplot(x='Class', y='Accuracy',  data=pdf,ci=0) 

ds_max = np.max(num_instances_per_cls)

y = num_instances_per_cls / ds_max * 100 #to scale to diagram size

plt.plot(x,y,color='black')

x1 = np.arange(num_cls) 

x2 = [str(i) for i in x1]

plt.xticks(x1,x2)

plt.savefig(save_file,bbox_inches='tight')

plt.show()






































