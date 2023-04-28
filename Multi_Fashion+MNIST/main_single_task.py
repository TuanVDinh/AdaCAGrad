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
from model_lenet import MultiLeNet, LeNet
import wandb
import copy
"""
Define task metrics, loss functions and model trainer here.
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

''' ===== multi task MGD trainer ==== '''
def multi_task_mgd_trainer(train_loader, test_loader, multi_task_model, device,
                           optimizer, scheduler, opt,
                           total_epoch=200, method='sumloss', alpha=0.5, seed=0,dataset_name=None):
    start_time = time.time()
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    acc_task_all, loss_task_all = [], []
    best_acc_val = -1
    for index in range(total_epoch):
        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        total = 0.0
        losses = 0.0
        for k in range(train_batch):
            
            img,ys = train_dataset.next()
            img, ys = img.to(device), ys.long().to(device)
            bs = len(ys)
            out = multi_task_model(img)
            if method == "single_L":
                train_loss_tmp = nn.CrossEntropyLoss()(out,ys[:, 0])     
            else:
                train_loss_tmp = nn.CrossEntropyLoss()(out,ys[:, 1])   
            optimizer.zero_grad()
            
            train_loss_tmp.backward()
            optimizer.step()
            losses_batch = train_loss_tmp.item()
            losses += bs * np.array([losses_batch])
            total += bs
        l_train = losses[0] / total
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
                bs = len(ys)
                # acc
                pred = out.data.max(1)[1]  # first column has actual prob.
                if method == "single_L":
                    task_correct += pred.eq(ys[:, 0]).sum()    
                else:
                    task_correct += pred.eq(ys[:, 1]).sum()   
                
                total += bs
            mean_acc = task_correct.cpu().item() / total
            print('Epoch: {:04d}, Acc_val_task: {:.4f}, Loss_train_task: {:.4f}'.format(index,mean_acc,l_train))
            if mean_acc > best_acc_val:
                print("Save model")
                best_acc_val = mean_acc
                torch.save(multi_task_model.state_dict(), f"models/{dataset_name}_{method}-{seed}.pt")
            acc_task_all.append(mean_acc)
            loss_task_all.append(l_train)
    avg_acc = np.mean(np.array(acc_task_all))
    avg_loss = np.mean(np.array(loss_task_all))

    print('AVG_acc_val_task: {:.4f},AVG_loss_train_task: {:.4f}'.format(avg_acc,avg_loss))
    end_time = time.time()
    print("Training time: ", end_time-start_time)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Multi-task: Split')
    parser.add_argument('--dataname', default='multi_fashion', type=str, help='multi_mnist, multi_fashion, multi_fashion_and_mnist')
    parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--dataroot', default='/home/tuantran/CAGrad/Ada_CAGrad/data/', type=str, help='dataset root')
    parser.add_argument('--method', default='single_L', type=str, help='optimization method')
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
    #wandb.init(project="Single_"+opt.dataname,name = opt.method, entity="tuantran23012000")
    # define model, optimiser and scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = LeNet().to(device)
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=0.05,weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print('Parameter Space: ABS: {:.1f}'.format(count_parameters(SegNet_MTAN)))

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
    multi_task_mgd_trainer(train_loader,
                    val_loader,
                    SegNet_MTAN,
                    device,
                    optimizer,
                    scheduler,
                    opt,
                    50,
                    opt.method,
                    opt.alpha, opt.seed,opt.dataname)