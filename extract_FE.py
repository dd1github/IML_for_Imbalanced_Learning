#This code is adapted from https://github.com/kaidic/LDAM-DRW

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from imbalance_cifar import IMBALANCECIFAR10
import resnet_XAI as models
import pandas as pd

from utils import *

t00 = time.time()
t0 = time.time()


torch.set_printoptions(precision=4, threshold=20000, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

print('model names ', model_names)

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar100', help='dataset setting')

parser.add_argument('-a', '--arch', metavar='ARCH',
                    default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet32)')

parser.add_argument('--num_cls', 
                    default=10, 
                    type=int, metavar='N',
                    help='number of classes)')

parser.add_argument('--model_path',
    default=".../CEbal_133_None_res32_best.pth",
                    type=str, help='path to the trained model')

parser.add_argument('--data_root',
    default=".../data/",
                    type=str, help='path to where data is stored')

parser.add_argument('--feat_file',
    default=".../CE_cifar10_train.csv",
    type=str, help='path to save the extracted feature embeddings')

parser.add_argument('--logits_file',
    default=".../cifar10_train_logits.csv",
    type=str, help='path to save the extracted logits')

parser.add_argument('--cls_accur_file',
    default=".../cifar10_train_acc.csv",
    type=str, help='path to save the extracted class accuracies')



parser.add_argument('--loss_type',
                    default="CE",
                    type=str, help='loss type')

parser.add_argument('--imb_type', default="exp",
                    type=str, help='imbalance type')

parser.add_argument('--imb_factor', 
                    default=0.01,
                    type=float, help='imbalance factor')

parser.add_argument('--train_rule',
                    default='None',
                    type=str,
                    help='data sampling strategy for train loader')

parser.add_argument('--rand_number', default=0, type=int,
                    help='fix random number for data sampling')

parser.add_argument('--exp_str', default='0', type=str,
                    help='number to indicate which experiment it is')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs',
                    default=1,
                    type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU id to use.')

parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed(0)


best_acc1 = 0
best_acc = 0  


args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))
print()

args.store_name = '_'.join([args.dataset, args.arch, args.loss_type,
                            args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])

use_norm = True if args.loss_type == 'LDAM' else False

model = models.__dict__[args.arch](num_classes=args.num_cls, use_norm=use_norm)

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)


model.load_state_dict(torch.load(args.model_path))


epoch = args.epochs

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = IMBALANCECIFAR10(root=args.data_root,
            imb_type=args.imb_type, imb_factor=args.imb_factor,
            rand_number=args.rand_number, train=True, download=True,
            transform=transform_val)


cls_num_list = train_dataset.get_cls_num_list()

train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)  



if args.train_rule == 'None':
    train_sampler = None
    per_cls_weights = None

if args.loss_type == 'CE':
    criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)


def validate(val_loader, model, criterion, epoch, args, f,f1,f2):

    losses = AverageMeter('Loss', ':.4e')

    global best_acc
    
    correct = 0
    total = 0
    
    class_correct = list(0. for i in range(args.num_cls))
    class_total = list(0. for i in range(args.num_cls))

    count = 0
    train_on_gpu = torch.cuda.is_available()
    
    classes = np.arange(args.num_cls)

    model.eval()
    all_preds = []
    all_targets = []
    all_values = []
    all_feats = []
    all_out = []
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, out_fm, out1 = model(input)

            out1 = out1.detach().cpu().numpy()
            
            loss = criterion(output, target)

            m = nn.Softmax(dim=1)
            soft = m(output)
            values, pred = torch.max(soft, 1)
            
            losses.update(loss.item(), input.size(0))
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_values.extend(values.detach().cpu().numpy())
            all_feats.extend(out1)
            
            out_np = output.detach().cpu().numpy()
            all_out.extend(out_np)
            
            
            tar_np = target.detach().cpu().numpy()
            
            tar_len = len(tar_np)
            
            total += target.size(0)

            pred_np = pred.detach().cpu().numpy()
            
            if count == 0:
                y_true = np.copy(tar_np)
                y_pred = np.copy(pred_np)
            else:
                y_true = np.concatenate((y_true, tar_np), axis=None)
                y_pred = np.concatenate((y_pred, pred_np), axis=None)
            count += 1

            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            

            for i in range(tar_len):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        
        cls_acc = []
        
        if epoch % 1 == 0:
            for i in range(args.num_cls):
                if class_total[i] > 0:
                    print('Accuracy of %5s: %2d%% (%2d/%2d)' % (
                        classes[i], 100 * class_correct[i] / class_total[i],
                        np.sum(class_correct[i]), np.sum(class_total[i])))
                    cls_acc.append(100 * class_correct[i] / class_total[i])
                    
                else:
                    print(
                        'Accuracy of %5s: N/A (no training examples)' % (classes[i]))

            print('\n Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(class_correct) / np.sum(class_total),
                np.sum(class_correct), np.sum(class_total)))

            
        allp = pd.DataFrame(data=all_preds, columns=['pred'])
        allt = pd.DataFrame(data=all_targets, columns=['actual'])
        allv = pd.DataFrame(data=all_values, columns=['certainty'])
        allf = pd.DataFrame(all_feats)
        allcomb = pd.concat([allt, allp, allv, allf], axis=1)
        
        allcomb.to_csv(f, index=False)  
        
        pd_out = pd.DataFrame(all_out)
        pd_out.to_csv(f1,index=False)
        
        pacc = pd.DataFrame(cls_acc,columns=['Acc'])
        pacc.to_csv(f2,index=False)
        
        
        
        
        
        
        

#############################################################

validate(train_loader, model, criterion, 1, args, f=args.feat_file,
         f1=args.logits_file,f2=args.cls_accur_file)

