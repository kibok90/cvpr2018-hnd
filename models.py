import os
import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

path = 'train'
if not os.path.isdir(path):
    os.makedirs(path)

def init_truncated_normal(model, aux_str=''):
    if model is None: return None
    init_path = '{path}/{in_dim:d}_{out_dim:d}{aux_str}.pth' \
                .format(path=path, in_dim=model.in_features, out_dim=model.out_features, aux_str=aux_str)
    if os.path.isfile(init_path):
        model.load_state_dict(torch.load(init_path))
        print('load init weight: {init_path}'.format(init_path=init_path))
    else:
        if isinstance(model, nn.ModuleList):
            [truncated_normal(sub) for sub in model]
        else:
            truncated_normal(model)
        print('generate init weight: {init_path}'.format(init_path=init_path))
        torch.save(model.state_dict(), init_path)
        print('save init weight: {init_path}'.format(init_path=init_path))
    
    return model

def truncated_normal(model):
    std = math.sqrt(2./(model.in_features + model.out_features))
    if model.bias is not None:
        model.bias.data.zero_()
    model.weight.data.normal_(std=std)
    truncate_me = (model.weight.data > 2.*std) | (model.weight.data < -2.*std)
    while truncate_me.sum() > 0:
        model.weight.data[truncate_me] = torch.normal(std=std*torch.ones(truncate_me.sum()))
        truncate_me = (model.weight.data > 2.*std) | (model.weight.data < -2.*std)
    return model

class DeepLinearReLU(nn.Sequential):
    def __init__(self, dims, no_last_relu=False, init=True):
        super(DeepLinearReLU, self).__init__()
        
        for d in range(len(dims)):
            if d == 0: continue
            sub = nn.Linear(dims[d-1], dims[d])
            if init: sub = init_truncated_normal(sub, '_d{d:d}'.format(d=d))
            self.add_module('fc{d}'.format(d=d), sub)
            if no_last_relu and d == len(dims)-1: continue
            self.add_module('relu{d}'.format(d=d), nn.ReLU())

class FLModel(nn.Linear):
    def reset_parameters(self):
        init_truncated_normal(self)
    
    def forward(self, input, input_norm=False):
        if input_norm: input = F.normalize(input, p=2, dim=1)
        return super(FLModel, self).forward(input)

class TDModel(nn.ModuleList):
    def __init__(self, in_dim, out_dims, ns=False, input_norm=False, relu=False, softmax='n'):
        super(TDModel, self).__init__()
        
        self.input_norm = input_norm
        self.relu = relu
        self.softmax = softmax
        
        self += [nn.Linear(in_dim, out_dim+int(ns)) for out_dim in out_dims]
        
        self.in_features = in_dim
        self.out_features = sum(out_dims)
        if ns: self.out_features += len(out_dims)
        self.bias = None
        
        init_truncated_normal(self, '_td')
    
    def forward(self, input, m=-1):
        if self.input_norm: input = F.normalize(input, p=2, dim=1)
        if m < 0:
            return torch.cat([self.postprocess(sub(input)) for sub in self], dim=1)
        else:
            return self.postprocess(self[m](input))
    
    def postprocess(self, input):
        if self.relu:
            input = F.relu(input)
        if self.softmax == 'l':
            input = F.log_softmax(input, dim=1)
        elif self.softmax == 's':
            input = F.softmax(input, dim=1)
        return input

class LOOLoss(nn.Module):
    def __init__(self, T, opts):
        super(LOOLoss, self).__init__()
        
        self.gpu = opts.gpu
        self.loo = opts.loo if 'LOO' in opts.method else 0.
        self.label_smooth = opts.label_smooth
        self.kld_u_const = math.log(len(T['wnids']))
        self.relevant = [torch.from_numpy(rel) for rel in T['relevant']]
        self.labels_relevant = torch.from_numpy(T['labels_relevant'].astype(np.uint8))
        ch_slice = T['ch_slice']
        if opts.class_wise:
            num_children = T['num_children']
            num_supers = len(num_children)
            self.class_weight = torch.zeros(ch_slice[-1])
            for m, num_ch in enumerate(num_children):
                self.class_weight[ch_slice[m]:ch_slice[m+1]] = 1. / (num_ch * num_supers)
        else:
            self.class_weight = torch.ones(ch_slice[-1]) / ch_slice[-1]
    
    def forward(self, input, target): # input = Variable(logits), target = labels
        loss = Variable(torch.zeros(1).cuda()) if self.gpu else Variable(torch.zeros(1))
        
        # novel loss
        if self.loo > 0.:
            target_novel = self.labels_relevant[target]
            for i, rel in enumerate(self.relevant):
                if target_novel[:,i].any():
                    relevant_loc = target_novel[:,i].nonzero().view(-1)
                    loss += -F.log_softmax(input[relevant_loc][:, rel], dim=1)[:,0].mean() * self.class_weight[i]
            loss *= self.loo
        
        # known loss
        log_probs = F.log_softmax(input, dim=1)
        loss += F.nll_loss(log_probs, Variable(target))
        
        # regularization
        if self.label_smooth > 0.:
            loss -= (log_probs.mean() + self.kld_u_const) * self.label_smooth
        
        return loss

    def cuda(self, device=None):
        super(LOOLoss, self).cuda(device)
        self.relevant = [rel.cuda(device) for rel in self.relevant]
        self.labels_relevant = self.labels_relevant.cuda(device)
        return self

class TDLoss(nn.Module):
    def __init__(self, T, opts):
        super(TDLoss, self).__init__()
        
        self.gpu = opts.gpu
        self.label_smooth = opts.label_smooth
        self.ex_smooth = opts.ex_smooth if opts.method == 'TD' else 0.
        self.class_wise = opts.class_wise
        self.novel_score = opts.novel_score
        self.labels_ch = torch.from_numpy(T['labels_ch'])
        self.labels_in = torch.from_numpy(T['labels_in'].astype(np.uint8))
        self.labels_out = torch.from_numpy(T['labels_out'].astype(np.uint8))
        self.root = T['root'] - len(T['wnids_leaf'])
        self.num_children = T['num_children']
        self.ch_slice = T['ch_slice']
        self.kld_u_const = [math.log(num_ch) for num_ch in self.num_children]
    
    def forward(self, input, target, m): # input = Variable(logits), target = labels
        loss = Variable(torch.zeros(1).cuda()) if self.gpu else Variable(torch.zeros(1))
        
        # known loss
        log_probs = F.log_softmax(input, dim=1)
        num_inputs = 0
        for i_ch in range(self.num_children[m]):
            target_ch = self.labels_ch[:, self.ch_slice[m]+i_ch][target]
            num_inputs_ch = (target_ch >= 0).sum()
            if num_inputs_ch > 0:
                num_inputs += num_inputs_ch
                loss += F.nll_loss(log_probs, Variable(target_ch), size_average=self.class_wise, ignore_index=-1)
        
        # training model parameters of novel classes
        if self.novel_score:
            # novel loss
            if m != self.root and self.ex_smooth > 0.: # root does not have exclusive leaves
                target_out = self.labels_out[:,m][target]
                num_inputs_ch = target_out.sum()
                if num_inputs_ch > 0:
                    num_inputs += num_inputs_ch
                    target_ch = self.num_children[m]*target_out.long()
                    loss += F.nll_loss(log_probs, Variable(target_ch),
                                       size_average=self.class_wise, ignore_index=0) * self.ex_smooth
            
            # known & novel loss normalization
            if self.class_wise:
                loss /= self.num_children[m] + (m != self.root)
            elif num_inputs > 0:
                loss /= num_inputs
        
        # smoothing output of novel classes
        else:
            # known loss normalization
            if self.class_wise:
                loss /= self.num_children[m]
            elif num_inputs > 0:
                loss /= num_inputs
            
            # novel loss
            if m != self.root and self.ex_smooth > 0.: # root does not have exclusive leaves
                target_out = self.labels_out[:,m][target]
                if target_out.any():
                    loss -= (log_probs[target_out.nonzero().view(-1)].mean() + self.kld_u_const[m]) * self.ex_smooth
        
        # regularization
        if self.label_smooth > 0.:
            target_in = self.labels_in[:,m][target]
            if target_in.any():
                loss -= (log_probs[target_in.nonzero().view(-1)].mean() + self.kld_u_const[m]) * self.label_smooth
        return loss
    
    def cuda(self, device=None):
        super(TDLoss, self).cuda(device)
        self.labels_ch = self.labels_ch.cuda(device)
        self.labels_in = self.labels_in.cuda(device)
        self.labels_out = self.labels_out.cuda(device)
        return self
