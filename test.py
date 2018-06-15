import os
import time
import copy
import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ee = 1e-8
MAX_DIST = 127

def harmonic_mean(a, b): return 2.*a*b/(a+b) if a > 0. and b > 0. else 0.

class Identity(nn.Module):
    def forward(self, x):
        return x

def test(work_name, data_loader, model, T, opts, save_path, start_time=time.time()):

    model.eval()

    if opts.test_data_norm:
        novel_biases = np.concatenate([-(2.**np.arange(4.,-3.-ee,-1.)),
                                       np.arange(-0.1,0.2+ee,0.002),
                                       2.**np.arange(-2.,4.+ee,1.)])
    else:
        novel_biases = np.concatenate([-(2.**np.arange(4.,2.-ee,-1.)),
                                       np.arange(-2.5,5.+ee,0.05),
                                       2.**np.arange(3.,4.+ee,1.)])
    
    num_leaves = len(T['wnids_leaf'])
    num_classes = len(T['wnids'])
    num_labels = len(T['label_hnd']) # len(T['wnids_leaf']) + len(T['wnids_novel'])
    num_pts = novel_biases.shape[0]
    classes = {'known': np.arange(num_leaves),
               'novel': np.arange(num_leaves, num_labels),
               'super': np.arange(num_leaves, num_classes)}
    dtypes = ['val'] if work_name == 'val' else ['known', 'novel']
    
    hierarchical_measure = T.get('dist_mat') is not None
    counters = init_counters(num_pts, num_classes if work_name == 'val' else num_labels, hierarchical_measure)
    results = init_results(num_pts, None, hierarchical_measure)
    labels = {'known': [], 'novel': []}
    preds  = {'known': [], 'novel': []}
    
    novel_classes = torch.LongTensor(np.arange(num_leaves, num_classes))
    if opts.gpu: novel_classes = novel_classes.cuda()
    
    for dtype in dtypes:
        num_data = len(data_loader[dtype].dataset)
        labels[dtype] = -np.ones(num_data, dtype=int)
        preds[dtype]  = -np.ones([num_pts, num_data], dtype=int)
        for i, (inputs, targets) in enumerate(data_loader[dtype]):
            inputs = Variable(inputs, volatile=True)
            if opts.gpu: inputs, targets = inputs.cuda(), targets.cuda()
            
            logits = model(inputs)
            
            pos = i*opts.batch_size
            labels[dtype][pos:pos+targets.size(0)] = targets.cpu().numpy()
            
            for b, novel_bias in enumerate(novel_biases):
                # flatten prediction
                logits_biased = logits.clone()
                logits_biased[:, novel_classes] += novel_bias
                preds_ = logits_biased.max(dim=1)[1].data.cpu().numpy()
                preds[dtype][b, pos:pos+targets.size(0)] = preds_
                
                if work_name == 'val':
                    # known performance evaluation with full taxonomy
                    count_test(b, counters, preds_, targets, T, hierarchical_measure)
                    
                    # novel performance evaluation with deficient taxonomy
                    count_val(b, counters, logits_biased, targets, T, opts.gpu, hierarchical_measure)
                else:
                    # performance evaluation with full taxonomy
                    count_test(b, counters, preds_, targets, T, hierarchical_measure)
        
        # print('{dtype} eval {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))
        
    # acc
    counters_to_results(counters, results, classes)
    results_to_guarantee(results, novel_biases, opts.known_guarantee)
    print_results(work_name, results, save_path, hierarchical_measure, start_time)
    
    if opts.save_results:
        counters['novel_biases'] = novel_biases
        results['novel_biases'] = novel_biases
        preds['novel_biases'] = novel_biases
        np.save('{save_path}_{work_name}_counters.npy'.format(save_path=save_path, work_name=work_name), counters)
        np.save('{save_path}_{work_name}_results.npy'.format(save_path=save_path, work_name=work_name), results)
        np.save('{save_path}_{work_name}_preds_all.npy'.format(save_path=save_path, work_name=work_name), preds)
    
    if work_name == 'test':
        np.save('{save_path}_{work_name}_labels.npy'.format(save_path=save_path, work_name=work_name), labels)
        preds_opt = {'known': preds['known'][results['acc']['loc']], 'novel': preds['novel'][results['acc']['loc']]}
        np.save('{save_path}_{work_name}_preds.npy'.format(save_path=save_path, work_name=work_name), preds_opt)

def val_td(data_loader, model, T, opts, save_path, start_time=time.time()):

    model.eval()
    
    work_name = 'val'
    
    ths = np.concatenate([[0.], 2.**np.arange(-10,2.+ee,.5)])
    
    num_leaves = len(T['wnids_leaf'])
    num_classes = len(T['wnids'])
    num_labels = len(T['label_hnd']) # len(T['wnids_leaf']) + len(T['wnids_novel'])
    num_pts = ths.shape[0]
    classes = {'known': np.arange(num_leaves), 'novel': np.arange(num_leaves, num_labels)}
    dtypes = ['val']
    
    num_models = len(model)
    num_children = T['num_children']
    multi_inds = T['multi_inds']
    classifiable = T['classifiable']
    
    hierarchical_measure = T.get('dist_mat') is not None
    counters = init_counters(num_pts, num_children, hierarchical_measure)
    results = init_results(num_pts, num_models, hierarchical_measure)
    
    for dtype in dtypes:
        for i, (inputs, targets) in enumerate(data_loader[dtype]):
            inputs = Variable(inputs, volatile=True)
            if opts.gpu: inputs, targets = inputs.cuda(), targets.cuda()
            
            logits = [[]]*num_models
            kld_u = [[]]*num_models
            
            for m, sub in enumerate(model):
                # filter classifiable labels
                b_relevant = torch.zeros(targets.size(), out=torch.ByteTensor())
                if opts.gpu: b_relevant = b_relevant.cuda()
                for k in classifiable[m]:
                    b_relevant |= (targets == k)
                if not b_relevant.any(): continue
                relevant = b_relevant.nonzero().view(-1)
                relevant_labels = targets[relevant].cpu().numpy()
                
                # feedforward
                logits[m] = sub(inputs[relevant])
                if opts.test_relu: logits[m] = F.relu(logits[m])
                if opts.test_data_norm: logits[m] = F.normalize(logits[m], p=2, dim=1)
                kld_u[m] = - math.log(num_children[m]) \
                           - F.log_softmax(logits[m], dim=1).data.sum(dim=1) / num_children[m]
                logits[m] = logits[m].data.cpu().numpy()
                kld_u[m] = kld_u[m].cpu().numpy()
                
                for t, th in enumerate(ths):
                    # prediction in super class
                    b_conf = kld_u[m] > th
                    # assert (kld_u[m] > -1e-4).all(), 'some kld negative'
                    preds_ = -1*np.ones_like(relevant_labels)
                    if b_conf.any():
                        preds_[b_conf] = logits[m][b_conf].argmax(axis=1)
                    
                    # performance evaluation
                    count_super(t, m, counters, preds_, relevant_labels, multi_inds[m])
        
        # print('{dtype} eval {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))
    
    counters_to_results_super(counters, results, ths, num_models)
    print_results_super(work_name, results, save_path, start_time)
    
    if opts.save_results:
        counters['ths'] = ths
        results['ths'] = ths
        np.save('{save_path}_{work_name}_counters.npy'.format(save_path=save_path, work_name=work_name), counters)
        np.save('{save_path}_{work_name}_results.npy'.format(save_path=save_path, work_name=work_name), results)
    
    return results['ths_opt']

def test_td(ths_opt, data_loader, model, T, opts, save_path, start_time=time.time()):

    model.eval()
    
    work_name = 'test'
    
    num_leaves = len(T['wnids_leaf'])
    num_classes = len(T['wnids'])
    num_labels = len(T['label_hnd']) # len(T['wnids_leaf']) + len(T['wnids_novel'])
    classes = {'known': np.arange(num_leaves), 'novel': np.arange(num_leaves, num_labels)}
    dtypes = ['known', 'novel']
    
    num_models = len(model)
    num_children = T['num_children']
    children = [np.array(ch) for ch in T['children']]
    root = T['root']
    
    if not isinstance(ths_opt, (np.ndarray, list)):
        ths_opt = ths_opt*np.ones(num_models)
    
    hierarchical_measure = T.get('dist_mat') is not None
    counters = init_counters(1, num_labels, hierarchical_measure)
    results = init_results(1, None, hierarchical_measure)
    labels = {'known': [], 'novel': []}
    preds  = {'known': [], 'novel': []}
    
    cls_order = [root]
    m = 0
    while m < len(cls_order):
        for ch in children[cls_order[m]]:
            if ch >= num_leaves and ch not in cls_order:
                cls_order.append(ch)
        m += 1
    assert len(cls_order) == num_models, 'some classes are unvisitable'
    
    for dtype in dtypes:
        num_data = len(data_loader[dtype].dataset)
        labels[dtype] = -np.ones(num_data, dtype=int)
        preds[dtype]  = -np.ones(num_data, dtype=int)
        for i, (inputs, targets) in enumerate(data_loader[dtype]):
            inputs = Variable(inputs, volatile=True)
            if opts.gpu: inputs, targets = inputs.cuda(), targets.cuda()
            
            pos = i*opts.batch_size
            labels[dtype][pos:pos+targets.size(0)] = targets.cpu().numpy()
            
            logits = [[]]*num_models
            kld_u = [[]]*num_models
            
            # feedforward
            for m, sub in enumerate(model):
                logits[m] = sub(inputs)
                if opts.test_relu: logits[m] = F.relu(logits[m])
                if opts.test_data_norm: logits[m] = F.normalize(logits[m], p=2, dim=1)
                kld_u[m] = - math.log(num_children[m]) \
                           - F.log_softmax(logits[m], dim=1).data.sum(dim=1) / num_children[m]
                logits[m] = logits[m].data.cpu().numpy()
                kld_u[m] = kld_u[m].cpu().numpy()
            
            # top-down prediction
            preds_ = root*np.ones_like(targets.cpu().numpy())
            for k in cls_order:
                b_relevant = (preds_ == k)
                if not b_relevant.any(): continue
                m = k - num_leaves
                b_conf = kld_u[m][b_relevant] > ths_opt[m]
                # assert (kld_u[m][b_relevant] > -1e-4).all(), 'some kld negative'
                if not b_conf.any(): continue
                relevant = b_relevant.nonzero()[0][b_conf]
                preds_[relevant] = children[k][logits[m][relevant].argmax(axis=1)]
            preds[dtype][pos:pos+targets.size(0)] = preds_
            
            # performance evaluation with full taxonomy
            count_test(0, counters, preds_, targets, T, hierarchical_measure)
        
        # print('{dtype} eval {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))
        
    # acc
    counters_to_results(counters, results, classes)
    results_to_guarantee(results, [ths_opt[0]], opts.known_guarantee)
    print_results(work_name, results, save_path, hierarchical_measure, start_time)
    
    if opts.save_results:
        counters['ths'] = ths_opt
        results['ths'] = ths_opt
        preds['ths'] = ths_opt
        np.save('{save_path}_{work_name}_counters.npy'.format(save_path=save_path, work_name=work_name), counters)
        np.save('{save_path}_{work_name}_results.npy'.format(save_path=save_path, work_name=work_name), results)
        np.save('{save_path}_{work_name}_preds_all.npy'.format(save_path=save_path, work_name=work_name), preds)
    
    if work_name == 'test':
        np.save('{save_path}_{work_name}_labels.npy'.format(save_path=save_path, work_name=work_name), labels)
        preds_opt = {'known': preds['known'][results['acc']['loc']], 'novel': preds['novel'][results['acc']['loc']]}
        np.save('{save_path}_{work_name}_preds.npy'.format(save_path=save_path, work_name=work_name), preds_opt)

def init_results(num_pts, num_classes=None, hierarchical_measure=False):

    if num_classes is None:
        acc_float = np.zeros(num_pts, dtype=float)
        auc_float = 0.
    else:
        acc_float = np.zeros([num_pts, num_classes], dtype=float)
        auc_float = np.zeros(num_classes, dtype=float)
    
    acc = {'known': acc_float, 'novel' : copy.deepcopy(acc_float),
           'harmonic': copy.deepcopy(acc_float),
           'auc'  : auc_float, 'g_bias': copy.deepcopy(auc_float),
           'g_known': copy.deepcopy(auc_float), 'g_novel': copy.deepcopy(auc_float),
           'g_harmonic': copy.deepcopy(auc_float),
          }
    results = {'acc': acc}
    if hierarchical_measure:
        results.update({'HP': copy.deepcopy(acc), 'HR': copy.deepcopy(acc),
                        'HF': copy.deepcopy(acc), 'HE': copy.deepcopy(acc),
                       })
    
    return results

def init_counters(num_pts, num_classes, hierarchical_measure=False):

    # top-down val
    if isinstance(num_classes, list):
        counter_data  = np.array( [np.zeros(num_ch+1, dtype=int) for num_ch in num_classes])
        counter_int   = np.array([[np.zeros(num_ch+1, dtype=int) for num_ch in num_classes]
                                  for _ in range(num_pts)])
        counter_float = np.array([[np.zeros(num_ch+1, dtype=float) for num_ch in num_classes]
                                  for _ in range(num_pts)])
    # top-down test or flatten
    else:
        counter_data  = np.zeros(num_classes, dtype=int)
        counter_int   = np.zeros([num_pts, num_classes], dtype=int)
        counter_float = np.zeros([num_pts, num_classes], dtype=float)
    
    counters = {'data': counter_data, 'acc' : counter_int}
    if hierarchical_measure:
        counters.update({'HP': counter_float,                'HR': copy.deepcopy(counter_float),
                         'HF': copy.deepcopy(counter_float), 'HE': copy.deepcopy(counter_int),
                        })
    
    return counters

def count_val(p, counters, logits, labels, T, gpu, hierarchical_measure=False):

    num_leaves = len(T['wnids_leaf'])
    num_classes = len(T['wnids'])
    num_supers = num_classes - num_leaves
    
    descendants = T['descendants']
    ch_slice = T['ch_slice']
    relevant = T['relevant']
    if hierarchical_measure:
        HP_mat = T['HP_mat']
        HF_mat = T['HF_mat']
        dist_mat = T['dist_mat']
    
    for m in range(num_supers):
        c = m + num_leaves
        for i in range(ch_slice[m], ch_slice[m+1]):
        
            # filter classifiable labels
            classify_me = torch.zeros(labels.size(), out=torch.ByteTensor())
            if gpu: classify_me = classify_me.cuda()
            for k in descendants[c]:
                classify_me |= (labels == k)
            if not classify_me.any(): continue
            
            loc = torch.LongTensor(relevant[i])
            if gpu: loc = loc.cuda()
            preds_c = logits[classify_me.nonzero().view(-1)][:, loc].max(dim=1)[1].data.cpu().numpy()
            
            acc = (preds_c == 0) # 0-th label is GT
            if hierarchical_measure:
                HP = HP_mat[relevant[i][preds_c], relevant[i][0]]
                HR = HP_mat[relevant[i][0], relevant[i][preds_c]]
                HF = HF_mat[relevant[i][preds_c], relevant[i][0]]
                HE = dist_mat[relevant[i][preds_c], relevant[i][0]]
            
            if p == 0: counters['data'][c] += preds_c.shape[0]
            counters['acc'][p,c] += acc.sum()
            if hierarchical_measure:
                counters['HP'][p,c] += HP.sum()
                counters['HR'][p,c] += HR.sum()
                counters['HF'][p,c] += HF.sum()
                counters['HE'][p,c] += HE.sum()

def count_test(p, counters, preds, labels, T, hierarchical_measure=False):

    label_hnd = T['label_hnd']
    
    if hierarchical_measure:
        HP_mat = T['HP_mat']
        HF_mat = T['HF_mat']
        dist_mat = T['dist_mat']
    
    for l in np.unique(labels.cpu().numpy()):
        preds_l = preds[(labels == int(l)).cpu().numpy().astype(bool)]
        acc = np.zeros_like(preds_l, dtype=bool)
        if hierarchical_measure:
            HE = MAX_DIST*np.ones_like(preds_l, dtype=int)
            HP, HR, HF = np.zeros_like(preds_l), np.zeros_like(preds_l), np.zeros_like(preds_l)
        for c in label_hnd[l]:
            acc |= (preds_l == c)
            if hierarchical_measure:
                HE = np.minimum(HE, dist_mat[preds_l, c])
                HP = np.maximum(HP, HP_mat[preds_l, c])
                HR = np.maximum(HR, HP_mat[c, preds_l])
                HF = np.maximum(HF, HF_mat[preds_l, c])
        
        if p == 0: counters['data'][l] += preds_l.shape[0]
        counters['acc'][p,l] += acc.sum()
        if hierarchical_measure:
            counters['HE'][p,l] += HE.sum()
            counters['HP'][p,l] += HP.sum()
            counters['HR'][p,l] += HR.sum()
            counters['HF'][p,l] += HF.sum()

def count_super(p, m, counters, preds, labels, label_to_ch):
    
    for l in np.unique(labels):
        preds_l = preds[labels == l]
        
        # in -> known
        if label_to_ch[l]:
            acc = np.zeros_like(preds_l, dtype=bool)
            for c in label_to_ch[l]:
                if p == 0: counters['data'][m][c] += preds_l.shape[0]
                acc |= (preds_l == c)
            acc_sum = acc.sum()
            for c in label_to_ch[l]:
                counters['acc'][p,m][c] += acc_sum
        
        # out -> novel
        else:
            if p == 0: counters['data'][m][-1] += preds_l.shape[0]
            acc_sum = (preds_l < 0).sum()
            counters['acc'][p,m][-1] += acc_sum

def counters_to_results(counters, results, classes, label_hnd=None):

    nonempty = (counters['data'] > 0).nonzero()[0]
    testable = dict()
    for dtype in ['known', 'novel']:
        testable[dtype] = classes[dtype][np.isin(classes[dtype], nonempty)]
    if label_hnd is None:
        num_data = dict()
        for dtype in ['known', 'novel']:
            num_data[dtype] = counters['data'][testable[dtype]]
    else:
        label_hnd_inv = dict()
        for dtype in ['known', 'novel']:
            for l in testable[dtype]:
                for c in label_hnd[l]:
                    if label_hnd_inv.get(c) is None:
                        label_hnd_inv[c] = [l]
                    else:
                        label_hnd_inv[c].append(l)
    
    for mtype in results:
        # acc
        for dtype in ['known', 'novel']:
            for p in range(results[mtype][dtype].shape[0]):
                if label_hnd is None: # leaf-wise
                    results[mtype][dtype][p] = (counters[mtype][p][testable[dtype]] / num_data[dtype]).mean()
                else: # super-wise; del
                    dtype_ = 'super' if dtype == 'novel' else dtype
                    num_classes_testable = 0
                    for c in classes[dtype_]:
                        if label_hnd_inv.get(c) is not None:
                            num_classes_testable += 1
                            l = label_hnd_inv[c]
                            results[mtype][dtype][p] += (counters[mtype][p][l] / counters['data'][l]).mean()
                    results[mtype][dtype][p] /= num_classes_testable
        
        # harmonic acc
        for p in range(results[mtype]['harmonic'].shape[0]):
            results[mtype]['harmonic'][p] = harmonic_mean(results[mtype]['known'][p],
                                                          results[mtype]['novel'][p])
        
        # AUC
        y = np.concatenate([[0.], results[mtype]['novel'],
                            [results[mtype]['novel'][-1]]])
        x = np.concatenate([[results[mtype]['known'][0]],
                            results[mtype]['known'], [0.]])
        results[mtype]['auc'] = -np.trapz(y,x);

def counters_to_results_super(counters, results, pts, num_models):
    
    num_pts = pts.shape[0]
    
    for m in range(num_models):
        b_known_testable = counters['data'][m] > 0
        b_novel_testable = b_known_testable[-1]
        b_known_testable[-1] = False
        b_known_testable_any = b_known_testable.any()
        
        for p in range(num_pts):
            if b_known_testable_any:
                # results['acc']['known'][p,m] = \
                    # (counters['acc'][p,m][b_known_testable] / counters['data'][m][b_known_testable]).mean()
                results['acc']['known'][p,m] = counters['acc'][p,m][:-1].sum() / counters['data'][m][:-1].sum()
            if b_novel_testable:
                results['acc']['novel'][p,m] = counters['acc'][p,m][-1] / counters['data'][m][-1]
            
            if b_known_testable_any and b_novel_testable:
                results['acc']['harmonic'][p,m] = harmonic_mean(results['acc']['known'][p,m],
                                                                results['acc']['novel'][p,m])
            elif b_known_testable_any:
                results['acc']['harmonic'][p,m] = results['acc']['known'][p,m]
            elif b_novel_testable:
                results['acc']['harmonic'][p,m] = results['acc']['novel'][p,m]
    
    # find optimal points
    i_opt = {'global': 0, 'local': []}
    results['acc_opt'] = {'global': {'known': [], 'novel': [], 'harmonic': []},
                          'local' : {'known': [], 'novel': [], 'harmonic': []}}
    results['ths_opt'] = {'global': 0., 'local': []}
    
    i_opt['global'] = results['acc']['harmonic'].mean(axis=1).argmax(axis=0)
    i_opt['local']  = results['acc']['harmonic'].argmax(axis=0)
    for mtype in ['known', 'novel', 'harmonic']:
        results['acc_opt']['global'][mtype] = results['acc'][mtype][i_opt['global']]
        results['acc_opt']['local'][mtype]  = results['acc'][mtype][i_opt['local'], range(num_models)]
    results['ths_opt']['global'] = pts[i_opt['global']]
    results['ths_opt']['local']  = pts[i_opt['local']]

def results_to_guarantee(results, pts, kg):

    # novel bias increasing -> known acc decreasing
    loc = np.abs(results['acc']['known'] - kg).argmin()
    closest = results['acc']['known'][loc]
    if closest < kg:
        locs = np.array([max(loc-1, 0), loc])
    elif closest == kg:
        locs = np.array([loc, loc])
    else:
        locs = np.array([loc, min(loc+1, results['acc']['known'].shape[0]-1)])
    
    # linear interpolation
    x = results['acc']['known'][locs]
    if x[0] == x[1]:
        for mtype in results:
            results[mtype]['g_known'] = results[mtype]['known'][loc]
            results[mtype]['g_novel'] = results[mtype]['novel'][loc]
            results[mtype]['g_harmonic'] = results[mtype]['harmonic'][loc]
        results['acc']['g_bias'] = pts[loc]
    else:
        for mtype in results:
            y = results[mtype]['known'][locs]
            results[mtype]['g_known'] = (kg-x[0])*(y[1]-y[0])/(x[1]-x[0])+y[0]
            y = results[mtype]['novel'][locs]
            results[mtype]['g_novel'] = (kg-x[0])*(y[1]-y[0])/(x[1]-x[0])+y[0]
            y = results[mtype]['harmonic'][locs]
            results[mtype]['g_harmonic'] = (kg-x[0])*(y[1]-y[0])/(x[1]-x[0])+y[0]
        y = pts[locs]
        results['acc']['g_bias'] = (kg-x[0])*(y[1]-y[0])/(x[1]-x[0])+y[0]
    results['acc']['loc'] = loc

def print_results(work_name, results, save_path, hierarchical_measure=False, start_time=time.time()):

    if hierarchical_measure:
        mtypes = ['acc', 'HF']
    else:
        mtypes = ['acc']
    
    print(save_path)
    print('{work_name}; '.format(work_name=work_name), end='')
    print('{time:8.3f} s'.format(time=time.time()-start_time))
    for m, mtype in enumerate(mtypes):
        print('bias: {res:7.4f}; '.format(res=results['acc']['g_bias']), end='')
        print('{mtype:4s}'.format(mtype=mtype), end='')
        print('known: {res:5.2f}; '.format(res=results[mtype]['g_known']*100.), end='')
        print('novel: {res:5.2f}; '.format(res=results[mtype]['g_novel']*100.), end='')
        if mtype == 'acc':
            print('auc  : {res:5.2f}; '.format(res=results[mtype]['auc']*100.))
        else:
            print('hmean: {res:5.2f}; '.format(res=results[mtype]['g_harmonic']*100.))
    
        # plot known vs. novel
        plt.figure(m)
        plt.plot(results[mtype]['known'], results[mtype]['novel'], 'k.-')
        if mtype == 'HE':
            plt.xticks(np.arange(0., 11., 1.))
            plt.yticks(np.arange(0., 11., 1.))
            plt.axis([0., 10., 0., 10.])
        else:
            plt.xticks(np.arange(0., 1.1, .1))
            plt.yticks(np.arange(0., 1.1, .1))
            plt.axis([0., 1., 0., 1.])
        plt.grid()
        plt.xlabel('known class accuracy')
        plt.ylabel('novel class accuracy')
        plt.title('known: {res:5.2f}; '.format(res=results[mtype]['g_known']*100.) + \
                  'novel: {res:5.2f}; '.format(res=results[mtype]['g_novel']*100.) + \
                  'hmean: {res:5.2f}; '.format(res=results[mtype]['g_harmonic']*100.) + \
                  'auc  : {res:5.2f}; '.format(res=results[mtype]['auc']*100.)
                 )
        plt.savefig(save_path + '_' + work_name + '_' + mtype + '.png')
        plt.clf()
        plt.close()

def print_results_super(work_name, results, save_path, start_time=time.time()):

    print(save_path)
    print('{work_name}; '.format(work_name=work_name), end='')
    print('global th: {th:6.4f}; '.format(th=results['ths_opt']['global']), end='')
    print('{time:8.3f} s'.format(time=time.time()-start_time))
    for ran in ['global', 'local']:
        ran_short = 'glb' if ran == 'global' else 'loc'
        
        print('{ran} acc '.format(ran=ran_short), end='')
        print('known: {known:5.2f}; '.format(known=results['acc_opt'][ran]['known'].mean()*100.), end='')
        print('novel: {novel:5.2f}; '.format(novel=results['acc_opt'][ran]['novel'].mean()*100.), end='')
        print('hmean: {harmonic:5.2f}'.format(harmonic=results['acc_opt'][ran]['harmonic'].mean()*100.))

if __name__ == '__main__':

    opts = argparse.Namespace()
    opts.gpu = True
    opts.dataset = 'ImageNet' #'ImageNet' #'AWA2' #'CUB'
    opts.cnn = 'resnet101'
    opts.test_relu = False
    opts.test_data_norm = False
    opts.workers = 0
    opts.batch_size = 5000 if opts.dataset == 'ImageNet' else 0
    opts.num_epochs = 50 if opts.dataset == 'ImageNet' else 5000
    opts.known_guarantee = 0.5
    opts.darts_path = 'train_darts/{dataset}/{cnn}'.format(dataset=opts.dataset, cnn=opts.cnn)
    opts.save_results = False
    
    if opts.dataset == 'AWA2':
        acc_guarantees = [99]
    else:
        acc_guarantees = [90]
    # acc_guarantees = [80, 85, 90, 95, 99]
    
    start_time = time.time()
    T = np.load('taxonomy/{dataset}/taxonomy.npy'.format(dataset=opts.dataset)).item()
    utils.update_taxonomy('LOO', T, -1, start_time)
    identity = Identity()
    
    for ag in acc_guarantees:
        # data loader
        dtypes = ['val', 'known', 'novel']
        opts.ag = ag
        data_loader = utils.get_feature_loader(dtypes, opts, start_time)
        
        save_path = '{darts_path}/{ag:.2f}'.format(darts_path=opts.darts_path, ag=ag/100.)
        print(save_path)
        
        test('val',  data_loader, identity, T, opts, save_path, start_time)
        test('test', data_loader, identity, T, opts, save_path, start_time)
