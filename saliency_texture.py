# -*- coding: utf-8 -*-


import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
from imbalance_cifar import IMBALANCECIFAR10
import numpy as np
import matplotlib.pyplot as plt
import resnet_XAI as models
import torchvision.datasets as datasets
from collections import defaultdict
import string

torch.set_printoptions(precision=4, threshold=60000, sci_mode=False)
np.set_printoptions(threshold=50000,precision=4,suppress=True)

#Illustration of texture visualization for CIFAR-10
###########################################################################
#inputs

#categories or archetypes: 0=safe, 1=border, 2=rare, 3=outlier
category = 3
categories = [0,1,2,3]

#percentage of pixels to treat as top-k or most salient
top_ratio = 0.1

#number of classes
num_cls = 10

#select a class label to visualize textures
#here, class 9 = trucks in CIFAR-10
cls_idx = [9] 

use_norm = False #only true for LDAM

#number of channels
C = 3

#height of input images
H = 32

#width of input images
W = 32

#number of color bins
num_bins = 14

#path to trained model
model_path= ".../CEbal_133_None_res32_best.pth"

#path to CIFAR-10 dataset
data_root = '.../data/' 

#save pdf of texture barchart
save_file=".../truck_textures.pdf"

##############################################################
#initial color bands or group bounds

d1 = 255.0

#black
blk_0 = np.array([0/d1,0/d1,0/d1]) 
blk_1 = np.array([77/d1,77/d1,77/d1]) 

#gray
gry_0 = np.array([50/d1,50/d1,50/d1]) 
gry_1 = np.array([200/d1,200/d1,200/d1]) 

#brown
brn_0 = np.array([100/d1,0/d1,0/d1]) 
brn_1 = np.array([150/d1,130/d1,100/d1]) 

#orange
or_0 = np.array([200/d1,100/d1,0/d1]) 
or_1 = np.array([255/d1,200/d1,150/d1])

#red
r_0 = np.array([150/d1,0/d1,0/d1]) 
r_1 = np.array([255/d1,150/d1,150/d1]) 

#green
g_0 = np.array([0/d1,150/d1,0/d1]) 
g_1 = np.array([150/d1,255/d1,150/d1]) 

#blue
b_0 = np.array([0/d1,0/d1,150/d1]) 
b_1 = np.array([150/d1,160/d1,255/d1])

#yellow
y_0 = np.array([160/d1,160/d1,0/d1]) 
y_1 = np.array([255/d1,255/d1,160/d1]) #b was 200

######################
#purple
p_0 = np.array([50/d1,0/d1,150/d1]) 
p_1 = np.array([200/d1,150/d1,255/d1]) 

#lavend1ar
lav_0 = np.array([150/d1,0/d1,200/d1]) 
lav_1 = np.array([255/d1,150/d1,255/d1]) 

#pink
pk_0 = np.array([150/d1,0/d1,50/d1]) 
pk_1 = np.array([255/d1,150/d1,200/d1])

#white
w_0 = np.array([190/d1,190/d1,190/d1])
w_1 = np.array([255/d1,255/d1,255/d1])


##############################################################
model = models.resnet32(num_classes=num_cls, use_norm=use_norm)

torch.cuda.set_device(0)
model = model.cuda(0)

model.load_state_dict(torch.load(model_path))

for param in model.parameters():
    param.requires_grad = False

train_dataset = IMBALANCECIFAR10(root=data_root,
        imb_type='exp', imb_factor=0.01,
        rand_number=0, train=True, download=True)#,

val_dataset = datasets.CIFAR10(root=data_root,
                train=False,download=True)

def preprocess(image):
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),        
    ])
    return transform(image)

inv_normalize = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010])

def toPIL(image):
    transform = T.Compose([
        T.ToPILImage(),
    ])
    return transform(image)

classes = train_dataset.classes
classes = [string.capwords(s) for s in classes]
classes

#targets
tars = train_dataset.targets

tars = np.array(tars)


##########################################################################
#find color textures in a specific class, using gradient

#cumulative tracking of bin counts, red, green and blue mus by bin
cum_bin_counts = np.zeros(num_bins,dtype=int)
cum_r_mus = np.zeros(num_bins,dtype=float)
cum_g_mus = np.zeros(num_bins,dtype=float)
cum_b_mus = np.zeros(num_bins,dtype=float)


ncg_total = 0


for n in cls_idx:
  ctar = tars[tars==n]
  cdata = train_dataset.data[tars==n]
  
  for i in range(len(cdata)):
    
    model.eval()
    model.zero_grad()
    
    X = preprocess(cdata[i])
    
    X = X.cuda(0, non_blocking=True)

    X = X.view(1,C,H,W)
       
    # gradient with respect to the input image
    X.requires_grad_()

    scores,_,_ = model(X)

    # Get the index corresponding to the maximum score and the maximum score itself.
    score_max_index = scores.argmax()
    
    score_max = scores[0,score_max_index]

    score_max.backward()

    saliency, ind = torch.max(X.grad.data.abs(),dim=1) 

    sorted1, indices = torch.sort(torch.flatten(saliency[0]))

    sorted1 = sorted1.detach().cpu().numpy()

    sorted_1 = sorted1[::-1]

    indices_1 = indices.detach().cpu().numpy()
    
    indices_1 = indices_1[::-1]

    slen = len(sorted1)

    topk = int(top_ratio * slen)
    #print('topk',topk)
    #topk #102

    top_ind = indices_1[:topk]

    X1 = torch.squeeze(X)
    
    with torch.no_grad():
        X1 = inv_normalize(X1)
    
    #red pixels
    r = X1[0]
    
    r = torch.flatten(r)
    
    r = r.detach().cpu().numpy()
    r = r[top_ind]
    
    #green pixels
    g = X1[1]
    
    g = torch.flatten(g)
    
    g = g.detach().cpu().numpy()
    g = g[top_ind]

    #blue pixels
    b = X1[2]
    
    b = torch.flatten(b)
    
    b = b.detach().cpu().numpy()
    b = b[top_ind]
    
    #################################################################
    d = defaultdict(list)
    
    #series of if statements to categorize pixels in color band bins
    for j in range(len(r)):
        if r[j] >= blk_0[0] and r[j] < blk_1[0] and \
           g[j] >= blk_0[1] and g[j] < blk_1[1] and \
           b[j] >= blk_0[2] and b[j] < blk_1[2]: 
                 d[0].append(j)
        
        elif r[j] >= brn_0[0] and r[j] < brn_1[0] and \
             g[j] >= brn_0[1] and g[j] < brn_1[1] and \
             b[j] >= brn_0[2] and b[j] < brn_1[2]:
                 d[4].append(j)
    
        elif r[j] >= or_0[0] and r[j] <= or_1[0] and \
             g[j] >= or_0[1] and g[j] < or_1[1] and \
             b[j] >= or_0[2] and b[j] < or_1[2]: 
                 d[5].append(j)
    
        elif r[j] >= r_0[0] and r[j] <= r_1[0] and \
             g[j] >= r_0[1] and g[j] < r_1[1] and \
             b[j] >= r_0[2] and b[j] < r_1[2]: 
                 d[6].append(j)
    
        elif r[j] >= g_0[0] and r[j] < g_1[0] and \
             g[j] >= g_0[1] and g[j] <= g_1[1] and \
             b[j] >= g_0[2] and b[j] < g_1[2]: 
                 d[7].append(j)
    
        elif r[j] >= b_0[0] and r[j] < b_1[0] and \
             g[j] >= b_0[1] and g[j] < b_1[1] and \
             b[j] >= b_0[2] and b[j] <= b_1[2]: 
                 d[8].append(j)
    
        elif r[j] >= y_0[0] and r[j] <= y_1[0] and \
             g[j] >= y_0[1] and g[j] <= y_1[1] and \
             b[j] >= y_0[2] and b[j] < y_1[2]: 
                 d[9].append(j)
    
        elif r[j] >= p_0[0] and r[j] < p_1[0] and \
             g[j] >= p_0[1] and g[j] < p_1[1] and \
             b[j] >= p_0[2] and b[j] <= p_1[2]: 
                 d[10].append(j)
    
        elif r[j] >= lav_0[0] and r[j] <= lav_1[0] and \
             g[j] >= lav_0[1] and g[j] < lav_1[1] and \
             b[j] >= lav_0[2] and b[j] <= lav_1[2]: 
                 d[11].append(j)
    
        elif r[j] >= pk_0[0] and r[j] <= pk_1[0] and \
             g[j] >= pk_0[1] and g[j] < pk_1[1] and \
             b[j] >= pk_0[2] and b[j] < pk_1[2]: 
                 d[12].append(j)
    
        elif r[j] >= w_0[0] and r[j] <= w_1[0] and \
             g[j] >= w_0[1] and g[j] <= w_1[1] and \
             b[j] >= w_0[2] and b[j] <= w_1[2]: 
                 d[13].append(j)
        
        
        ###############################
        #white
        elif r[j] >= .82 and \
             g[j] >= .82 and \
             b[j] >= .64: 
                 d[13].append(j)
        
        #catch-all light red
        elif r[j] >= .6 and g[j] >= .58 and b[j] >= .58 and \
             r[j] > g[j] and r[j] > b[j]:
                 d[12].append(j)
        
        #catch-all  blue
        elif r[j] <= .365 and g[j] <= .41 and b[j] >= .39 and \
             b[j] < .6: #> r[j] and b[j] > g[j]:
                 d[8].append(j)
             
        #catch-all - blue
        elif r[j] <= .21 and g[j] >= .4 and b[j] >= .43 and \
            g[j] <= .52 and \
            b[j] > r[j] and b[j] > g[j]:
                  d[8].append(j)
        
        #catch-all - blue
        elif r[j] <= .55 and g[j] >= .55 and b[j] >= .55 and \
            b[j] > g[j]:
                  d[8].append(j)
        
        #catch-all light blue
        elif r[j] >= .6 and g[j] >= .6 and b[j] >= .6 and \
             b[j] > r[j] and b[j] > g[j]:
                 d[10].append(j)
                 
        #catch-all light blue
        elif r[j] >= .05 and g[j] >= .5 and r[j] <= .75 and g[j] <= .75 and \
                b[j] >= .75 and b[j] <= 1.0 and \
                b[j] > r[j] and b[j] > g[j]:
                 d[10].append(j)     
                 
        #catch-all - light blue
        elif r[j] > .43 and g[j] >.5 and r[j] <= .85 and g[j] <= .85 and \
             b[j] > r[j] and b[j] > g[j] and \
             b[j] > .78 and b[j] <= 1.0:
                 d[10].append(j)
        
        #catch all light blue
        elif r[j] > .33 and g[j] >.5 and r[j] <= .61 and g[j] <= .9 and \
             b[j] > r[j] and b[j] > g[j] and \
             b[j] > .78 and b[j] <= 1.0:
                 d[10].append(j)
        
        #catch all light blue
        elif r[j] > .41 and g[j] >.5 and r[j] <= .61 and g[j] <= .9 and \
             b[j] > r[j] and b[j] > g[j] and \
             b[j] > .89:
                 d[10].append(j)       
        
        #catch all light blue
        elif r[j] > .55 and g[j] >.89 and r[j] <= .8 and \
             b[j] > r[j] and b[j] > g[j] and \
             b[j] > .93:
                 d[10].append(j)       
        
        
        #catch-all light green
        elif r[j] >= .49 and g[j] >= .6 and b[j] >= .2 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[11].append(j)
        
        #catch-all light green
        elif r[j] < .49 and g[j] >= .77 and b[j] >= .58 and b[j] < .75 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[11].append(j)
        
        
        #catch-all - brown
        elif r[j] <= .55 and g[j] <= .45 and b[j] <= .41 and \
             r[j] > g[j] and r[j] > b[j]:
                 d[4].append(j)
        
        #catch-all - brown
        elif r[j] >= .55 and g[j] >= .5 and b[j] <= .2 and \
             r[j] <= .8 and g[j] <= .75 and \
             r[j] > g[j] and r[j] > b[j]:
                 d[4].append(j)
        
        #catch-all - brown
        elif r[j] >= .5 and g[j] >= .5 and b[j] <= .2 and \
             r[j] <= .6 and g[j] <= .6 and \
             r[j] > g[j] and r[j] > b[j]:
                 d[4].append(j)
        
        #catch-all - grn
        elif r[j] <= .3 and g[j] >= .45 and b[j] >= .3 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[7].append(j)
        
        
        #catch-all - grn
        elif r[j] <= .4 and g[j] <= .53 and b[j] <= .4 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[7].append(j)
        
        #catch-all - grn
        elif r[j] <= .25 and g[j] >= .47 and b[j] <= .25 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[7].append(j)
        
        #catch-all - grn
        elif r[j] <= .63 and g[j] >= .5 and b[j] <= .2 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[7].append(j)
        
        #catch-all - grn
        elif r[j] >= .4 and g[j] >= .4 and b[j] <= .2 and \
             r[j] <= .63 and g[j]<= .75 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[7].append(j)
        
        
        #catch-all - drk grn
        elif r[j] <= .4 and g[j] <= .53 and b[j] <= .2 and \
             g[j] > r[j] and g[j] > b[j]:
                 d[1].append(j)
        
        #catch-all - drk blue
        elif r[j] <= .4 and g[j] <= .4 and b[j] <= .4 and \
             b[j] > r[j] and b[j] > g[j]:
                 d[2].append(j)
        
        #catch-all - drk blue
        elif r[j] <= .4 and g[j] <= .4 and b[j] >= .4 and b[j] <= .5 and \
             b[j] > r[j] and b[j] > g[j]:
                 d[2].append(j)
        
        # gray
        elif r[j] >= .7 and r[j] < .81 and \
             g[j] >= .7 and g[j] < .81 and \
             b[j] >= .7 and b[j] < .81: 
                 d[3].append(j)
       
        #moved gray to end
        elif r[j] >= gry_0[0] and r[j] < gry_1[0] and \
             g[j] >= gry_0[1] and g[j] < gry_1[1] and \
             b[j] >= gry_0[2] and b[j] < gry_1[2]: 
                 d[3].append(j)
       
    
    #################################################################
    
    bin_counts = np.zeros(num_bins,dtype=int)
    bin_mus = np.zeros(num_bins,dtype=float)

    r_mus = np.zeros(num_bins,dtype=float)
    g_mus = np.zeros(num_bins,dtype=float)
    b_mus = np.zeros(num_bins,dtype=float)

    for k in range(num_bins):
        bin_counts[k]=len(d[k])
        if bin_counts[k] > 0:
            rb = r[d[k]]
            gb = g[d[k]]
            bb = b[d[k]]
        
            r_mu = np.mean(rb)
            g_mu = np.mean(gb)
            b_mu = np.mean(bb)
        
            r_mus[k]= r_mu
            g_mus[k]= g_mu
            b_mus[k]= b_mu
            
            denom = (cum_bin_counts[k] + bin_counts[k])
            
            new_r = (cum_r_mus[k] * cum_bin_counts[k] + \
                r_mu * bin_counts[k]) / denom
            new_g = (cum_g_mus[k] * cum_bin_counts[k] + \
                g_mu * bin_counts[k]) / denom
            new_b = (cum_b_mus[k] * cum_bin_counts[k] + \
                b_mu * bin_counts[k]) / denom
    
            cum_r_mus[k]= np.copy(new_r)
            cum_g_mus[k]= np.copy(new_g)
            cum_b_mus[k]= np.copy(new_b)
            cum_bin_counts[k]+=bin_counts[k]
    
    
#################################################################
#################################################################
#display

colors = []

for i in range(num_bins):
    colors.append((cum_r_mus[i],cum_g_mus[i],cum_b_mus[i]))

colors

bin_labs = np.arange(num_bins)
values1 = np.copy(cum_bin_counts) 

values = values1 / np.sum(values1)

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(bin_labs, values, color = colors)

cls_idx_num = cls_idx[0]
cls_idx_num

#color bin labels
xlabs = ['Blk','Drk-G','Drk-B','Gray','Brn','Org','Red','Grn','Blue','Yel','Lt-B','Lt-G',
         'Lt-R','Wh']


plt.xlabel("Color Group Bins")
plt.ylabel("Percent for Class")

plt.title("%s: Salient Pixel Textures Used by Model to Predict Class" \
          %(classes[cls_idx_num]))

plt.xticks(np.arange(num_bins),xlabs)

plt.savefig(save_file,bbox_inches='tight')
    
plt.show()

######################################################################
######################################################################













