import numpy as np
import time
import torch
import torch.nn as nn
from copy import deepcopy
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize
import argparse
import torch.optim as optim
from utils import *
from data import Dataset
import random
from model_lenet import MultiLeNet,LeNet
import wandb
import copy
"""
Define task metrics, loss functions and model trainer here.
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

''' ===== multi task MGD trainer ==== '''
def test_mnist(train_loader, test_loader, multi_task_model, device,opt):
    test_batch = len(test_loader)
    total = 0.0
    losses = 0.0
    multi_task_model.eval()
    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(test_loader)
        task1_correct, task2_correct = 0.0, 0.0
        total = 0.0
        mean_acc = 0.0
        for k in range(test_batch):
            img,ys = test_dataset.next()
            img, ys = img.to(device), ys.long().to(device)
            task1, task2 = multi_task_model(img)
            train_loss_tmp = [nn.CrossEntropyLoss()(task1,ys[:, 0]),nn.CrossEntropyLoss()(task2,ys[:, 1])]
            bs = len(ys)

            # acc
            pred1 = task1.data.max(1)[1]  # first column has actual prob.
            pred2 = task2.data.max(1)[1]  # first column has actual prob.
            task1_correct += pred1.eq(ys[:, 0]).sum()
            task2_correct += pred2.eq(ys[:, 1]).sum()
            total += bs

            # loss
            losses_batch = [train_loss_tmp[0].item(),train_loss_tmp[1].item()]
            losses +=  np.array(losses_batch)
            #total += bs
        mean_acc = ((task1_correct.cpu().item() / total) + (task2_correct.cpu().item() / total))/2
        mean_loss = ((losses[0]/test_batch) + (losses[1]/test_batch))/2
        print('Mean_acc: {:.4f}, Acc_val_task1: {:.4f}, Acc_val_task2: {:.4f},Mean_loss: {:.4f}, Loss_train_task1: {:.4f}, Loss_train_task2: {:.4f}'.format(mean_acc,task1_correct.cpu().item() / total,task2_correct.cpu().item() / total,mean_loss,losses[0]/test_batch,losses[1]/test_batch))

def test_single_mnist(train_loader, test_loader, multi_task_model, device,opt):
    test_batch = len(test_loader)
    total = 0.0
    losses = 0.0
    multi_task_model.eval()
    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(test_loader)
        task_correct = 0.0
        total = 0.0
        mean_acc = 0.0
        for k in range(test_batch):
            img,ys = test_dataset.next()
            img, ys = img.to(device), ys.long().to(device)
            out = multi_task_model(img)
            train_loss_tmp = nn.CrossEntropyLoss()(out,ys[:, 1])
            bs = len(ys)

            # acc
            pred = out.data.max(1)[1]  # first column has actual prob.
            task_correct += pred.eq(ys[:, 1]).sum()
            total += bs

            # loss
            losses_batch = [train_loss_tmp.item()]
            losses +=  np.array(losses_batch)
            #total += bs
        mean_acc = task_correct.cpu().item() / total
        mean_loss = losses[0]/test_batch
        print('Mean_acc: {:.4f},Mean_loss: {:.4f}'.format(mean_acc,mean_loss))
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Multi-task: Split')
    parser.add_argument('--dataname', default='multi_fashion_and_mnist', type=str, help='multi_mnist, multi_fashion, multi_fashion_and_mnist')
    parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
    parser.add_argument('--method', default='cagrad', type=str, help='optimization method')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
    parser.add_argument('--alpha', default=0.2, type=float, help='the alpha')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.5, type=float, help='the learning rate')
    parser.add_argument('--seed', default=0, type=int, help='the seed')
    parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
    opt = parser.parse_args()
    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    #wandb.init(project="CAGrad_"+opt.dataname,name = opt.method, entity="tuantran23012000")
    # define model, optimiser and scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = MultiLeNet(n_tasks = 2).to(device)
    #SegNet_MTAN = LeNet().to(device)
    checkpoint = torch.load("/home/tuantran/CAGrad/Ada_CAGrad/models/multi_fashion_and_mnist_adacagrad-equal-10.0-0.pt")
    SegNet_MTAN.load_state_dict(checkpoint)
    #print('Parameter Space: ABS: {:.1f}'.format(count_parameters(SegNet_MTAN)))

    # define dataset
    path = opt.dataroot+opt.dataname+".pickle"
    val_size = 0.1
    bs = opt.bs
    data = Dataset(path, val_size=val_size)
    if val_size > 0:
        train_set, val_set, test_set = data.get_datasets()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=True, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=bs, shuffle=False, num_workers=0
        )
    else:
        train_set, val_set = data.get_datasets()
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=False, num_workers=2
        )
    # Train and evaluate multi-task network
    print(opt.method)
    test_mnist(train_loader,
                    test_loader,
                    SegNet_MTAN,
                    device,opt)