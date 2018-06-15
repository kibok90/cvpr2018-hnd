import os
import time
import copy
import argparse

import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.backends.cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import cnns, models, utils, test

ee = 1e-8

parser = argparse.ArgumentParser(description='Commands')
parser.add_argument('-gpu', '--gpu', action='store_true', help='GPU mode')
parser.add_argument('-w', '--workers', default=0, type=int, metavar='N',
                    help='Number of data loading workers (default: 0)')
parser.add_argument('-data', '--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('-cnn', '--cnn', default='resnet101', type=str, help='CNN name')
parser.add_argument('-test', '--test', action='store_true', help='Validation and test after training')
parser.add_argument('-res', '--save_results', action='store_true', help='Save validation and test results')
parser.add_argument('-m', '--method', default='LOO', type=str, help='Method; TD, RLB, LOO, TD+LOO, TD+RLB')
parser.add_argument('-nlr', '--no_last_relu', action='store_true', help='No ReLU at the last additional layer')
parser.add_argument('-tdn', '--test_data_norm', action='store_true', help='Data normalization for test only')

parser.add_argument('-tdname', '--td_name', default='', type=str, help='TD model hyperparams; TD+LOO, TD+RLB')
parser.add_argument('-r', '--radius', default=-1, type=int, help='Range of exclusive classes; -1 for inf; TD')
parser.add_argument('-rl', '--relabel', default=50, type=int, help='Relabeling rate; RLB, TD+RLB')
parser.add_argument('-loo', '--loo', default=1., type=float, help='LOO loss weight; LOO, TD+LOO')
parser.add_argument('-ex', '--ex_smooth', default=1., type=float, help='Exclusive class label smoothing weight; TD')
parser.add_argument('-lsm', '--label_smooth', default=0., type=float,
                    help='In-class label smoothing weight; TD, RLB, LOO, TD+LOO, TD+RLB')
parser.add_argument('-nl', '--num_layers', default=0, type=int, help='Number of additional FC+ReLU layers')
parser.add_argument('-ns', '--novel_score', action='store_true', help='Additional score for novel classes; TD')
parser.add_argument('-cw', '--class_wise', action='store_true', help='Class-wise loss; TD, LOO, TD+LOO')
parser.add_argument('-relu', '--test_relu', action='store_true', help='Apply ReLU at test time; TD, TD+LOO, TD+RLB')
parser.add_argument('-sm', '--softmax', default='l', type=str,
                    help='Apply softmax after TD; n: none, l: log softmax, s: softmax; TD+LOO')

parser.add_argument('-wd', '--wd', '--weight_decay', default=1e-2, type=float, help='Weight decay')
parser.add_argument('-lr', '--lr', '--learning_rate', default=1e-2, type=float, help='Initial learning rate')
parser.add_argument('-lrd', '--lr_decay', default=0.1, type=float, help='Learning rate decay')
parser.add_argument('-nlrd', '--num_lr_decay', default=2, type=int, help='Number of learning rate decays allowed')
parser.add_argument('-nep', '--num_epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('-bsize', '--batch_size', default=5000, type=int, help='Batch size')
parser.add_argument('-kg', '--known_guarantee', default=0.5, type=float, help='Known class accuracy guarantee')
parser.add_argument('-keep', '--keep', action='store_true', help='Keep past models')
parser.add_argument('-freq', '--save_freq', default=100, type=int, help='Save frequency when full-batch mode')

def main(opts, start_time=time.time()):

    # taxonomy
    T = np.load('taxonomy/{dataset}/taxonomy.npy'.format(dataset=opts.dataset)).item()
    utils.update_taxonomy(opts.method, T, opts.radius, start_time)
    
    # model
    data_dim = 2048 # feature dimension before softmax
    
    # top-down
    if opts.method == 'TD':
        model = models.TDModel(data_dim, T['num_children'], ns=opts.novel_score)
    
    # combined
    elif 'TD+' in opts.method:
        TDModel = models.TDModel(data_dim, T['num_children'],
                                 ns=opts.novel_score, relu=opts.test_relu, softmax=opts.softmax)
        FLModel = models.FLModel(sum(T['num_children']), len(T['wnids']))
        model = nn.Sequential(TDModel, FLModel)
    
    # flatten
    else:
        if opts.method == 'LOO' and opts.loo == 0.:
            model = models.FLModel(data_dim, len(T['wnids_leaf']))
        else:
            model = models.FLModel(data_dim, len(T['wnids']))
        
        # deep flatten
        if opts.num_layers > 0:
            model = nn.Sequential(
                        models.DeepLinearReLU([data_dim]*(opts.num_layers+1), no_last_relu=opts.no_last_relu),
                        model)
    
    if opts.gpu: model = model.cuda()
    
    torch.backends.cudnn.benchmark = True
    
    # optimizer and scheduler
    model_parameters = FLModel.parameters() if 'TD+' in opts.method else model.parameters()
    if opts.batch_size > 0:
        optimizer = SGD(model_parameters, lr=opts.lr, weight_decay=opts.wd, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opts.lr_decay, patience=0,
                                      verbose=False, threshold=2e-2, threshold_mode='rel',
                                      cooldown=0, min_lr=0, eps=1e-8)
    else: # full-batch
        optimizer = Adam(model_parameters, lr=opts.lr, weight_decay=opts.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opts.lr_decay, patience=10,
                                      verbose=False, threshold=1e-4, threshold_mode='rel',
                                      cooldown=0, min_lr=0, eps=1e-8)
        scheduler.cooldown_counter = opts.num_epochs // 10 # loss would increase in the first few epochs
    
    # loss
    loss_fn = models.TDLoss(T, opts) if opts.method == 'TD' else models.LOOLoss(T, opts)
    if opts.gpu: loss_fn = loss_fn.cuda()
    
    # save path
    save_path = utils.get_path(opts)
    print(save_path)
    
    # load the recent model
    epoch = utils.load_model(model, optimizer, scheduler, save_path, opts.num_epochs, start_time)
    if ('TD+' in opts.method) and epoch == 0:
        td_path = 'train/{dataset}/{cnn}/{method}/{td_name}' \
                  .format(dataset=opts.dataset, cnn=opts.cnn, method='TD', td_name=opts.td_name)
        utils.load_model(TDModel, None, None, td_path, opts.num_epochs, start_time)
    prev_epoch = 0 if opts.keep else epoch
    saved = True
    
    # data loader
    if opts.test: dtypes = ['train', 'val', 'known', 'novel']
    else:         dtypes = ['train']
    if 'data_loader' in opts:
        print('data_loader exists; {time:8.3f} s'.format(time=time.time()-start_time))
        data_loader = opts.data_loader
    else:
        data_loader = utils.get_feature_loader(dtypes, opts, start_time)
        opts.data_loader = data_loader
    
    
    # recover labels if relabeled previously
    if 'train' in data_loader and hasattr(data_loader['train'].dataset, 'target_tensor_bak'):
        data_loader['train'].dataset.target_tensor = data_loader['train'].dataset.target_tensor_bak
        del data_loader['train'].dataset.target_tensor_bak
        print('labels recovered')
    
    # relabel
    if 'RLB' in opts.method:
        dataset_path = 'datasets/{dataset}'.format(dataset=opts.dataset)
        relabels = utils.relabel(opts.relabel, data_loader['train'].dataset.target_tensor, T,
                                 opts.num_epochs, dataset_path, start_time)
        data_loader['train'].dataset.target_tensor_bak = data_loader['train'].dataset.target_tensor
    
    # min lr
    min_lr = opts.lr*(opts.lr_decay**opts.num_lr_decay) - ee
    
    print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch, num_epochs=opts.num_epochs), end='')
    print('start training; ', end='')
    print('{time:8.3f} s'.format(time=time.time()-start_time))
    for epoch in range(epoch+1, opts.num_epochs+1):
    
        # stopping criterion
        if optimizer.param_groups[0]['lr'] == 0.: break
        
        # train
        if 'RLB' in opts.method:
            data_loader['train'].dataset.target_tensor = relabels[epoch-1]
        
        loss_val = train(data_loader['train'], model, loss_fn, optimizer, epoch, T, opts, start_time)
        
        # lr decay
        if scheduler is not None:
            scheduler.step(loss_val)
            if optimizer.param_groups[0]['lr'] < min_lr:
                optimizer.param_groups[0]['lr'] = 0.
        
        # save model
        saved = False
        if opts.batch_size > 0 or epoch % opts.save_freq == 0:
            utils.save_model(model, optimizer, scheduler, save_path, epoch, opts.num_epochs, prev_epoch, start_time)
            if not opts.keep: prev_epoch = epoch
            saved = True
    
    if not saved:
        utils.save_model(model, optimizer, scheduler, save_path, epoch, opts.num_epochs, prev_epoch, start_time)
    
    print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch, num_epochs=opts.num_epochs), end='')
    print('training done; ', end='')
    print('{time:8.3f} s'.format(time=time.time()-start_time))
    
    # eval
    if opts.test:
        if opts.test_data_norm:
            save_path += '_tdn'
        if opts.method == 'TD':
            ths_opt = test.val_td(data_loader, model, T, opts, save_path, start_time)
            test.test_td(ths_opt['local'], data_loader, model, T, opts, save_path, start_time)
        else:
            test.test('val',  data_loader, model, T, opts, save_path, start_time)
            test.test('test', data_loader, model, T, opts, save_path, start_time)

def train(train_loader, model, loss_fn, optimizer, epoch, T, opts, start_time):

    loss_val = 0.
    num_iters = len(train_loader)
    num_prints = min([5, num_iters])
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = Variable(inputs.cuda()) if opts.gpu else Variable(inputs)
        targets = targets.cuda() if opts.gpu else targets
        
        optimizer.zero_grad()
        if opts.method == 'TD':
            batch_loss = Variable(torch.zeros(1).cuda()) if opts.gpu else Variable(torch.zeros(1))
            for m, sub in enumerate(model):
                logits = sub(inputs)
                batch_loss += loss_fn(logits, targets, m)
        else:
            logits = model(inputs)
            batch_loss = loss_fn(logits, targets)
        loss_val += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()
        
        print_me = (opts.batch_size > 0 and \
                    ((i+1) % (num_iters // num_prints) == 0 or i == 0 or i == num_iters-1)) \
                   or (opts.batch_size == 0 and (epoch == 1 or epoch % opts.save_freq == 0))
        if print_me:
            print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch, num_epochs=opts.num_epochs), end='')
            print('{iter:3d}/{num_iters:3d} i; '.format(iter=i+1, num_iters=num_iters), end='')
            print('lr: {lr:.0e}; '.format(lr=optimizer.param_groups[0]['lr']), end='')
            print('bl: {loss:9.3f}; '.format(loss=batch_loss.data[0]), end='')
            print('ml: {loss:9.3f}; '.format(loss=float(loss_val)/(i+1)), end='')
            print('{time:8.3f} s'.format(time=time.time()-start_time))
    
    return loss_val

if __name__ == '__main__':
    opts = parser.parse_args()
    print(opts)
    
    start_time = time.time()
    
    main(opts, start_time)
