import os
import time
import copy
import argparse

import scipy.io as sio
import h5py
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn
import torchvision.transforms as transforms

import cnns, folder, samplers

def extract_features(dtypes, opts, start_time=time.time()):
    cnn = cnns.__dict__[opts.cnn](pretrained=True)
    if opts.gpu:
        cnn = torch.nn.DataParallel(cnn).cuda()
        torch.backends.cudnn.benchmark = True
    cnn.eval()
    
    for dtype in dtypes:
        path = 'datasets/{dataset}'.format(dataset=opts.dataset)
        data_path = '{path}/{cnn}_{dtype}.h5'.format(path=path, cnn=opts.cnn, dtype=dtype)
        if not os.path.isfile(data_path):
            # dataset
            dataset = folder.StaticImageFolder(dataset=opts.dataset, dtype=dtype,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]))
            num_data = len(dataset)
            labels_ = np.array([dataset.imgs[k][1] for k in range(num_data)])
            
            # data loader
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opts.batch_size, shuffle=False,
                num_workers=opts.workers, pin_memory=True)
            print('{dtype}; data loader; '.format(dtype=dtype), end='')
            print('{time:8.3f} s'.format(time=time.time()-start_time))
            
            # feature extraction
            data_dim = 2048 # feature dimension before softmax
            data = torch.zeros(num_data, data_dim)
            labels = torch.zeros(num_data, out=torch.LongTensor())
            for i, (inputs, targets) in enumerate(data_loader):
                pos = i*opts.batch_size
                inputs = Variable(inputs.cuda(), volatile=True) if opts.gpu else Variable(inputs, volatile=True)
                data[pos:pos+inputs.size(0)] = cnn(inputs).data.cpu()
                labels[pos:pos+targets.size(0)] = targets
                print('{dtype}; '.format(dtype=dtype), end='')
                print('{pos:7d}/{num_data:7d} i; '.format(pos=pos, num_data=num_data), end='')
                print('{time:8.3f} s'.format(time=time.time()-start_time))
            data = data.numpy()
            labels = labels.numpy()            
            
            # sanity check
            print('{dtype}; '.format(dtype=dtype), end='')
            print('order of image path and data {consistency}consistent; ' \
                  .format(consistency='' if (labels_ == labels).all() else 'not '), end='')
            print('{time:8.3f} s'.format(time=time.time()-start_time))
            
            # save
            with h5py.File(data_path, 'w') as f:
                f.create_dataset('data',   data=data,   compression='gzip', compression_opts=9)
                f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
        
        print('{dtype}; {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))

def get_feature_loader(dtypes, opts, start_time=time.time()):

    data_loader = dict()
    
    for dtype in dtypes:
        data_path = 'datasets/{dataset}/{cnn}_{dtype}.h5'.format(dataset=opts.dataset, cnn=opts.cnn, dtype=dtype)
        with h5py.File(data_path, 'r') as f:
            if ('darts_path' in opts) and ('ag' in opts):
                darts_path = '{darts_path}/{dtype}_{ag:.2f}.mat' \
                             .format(darts_path=opts.darts_path, dtype=dtype, ag=opts.ag/100.) 
                inputs = torch.from_numpy(h5py.File(darts_path, 'r')['expected_rewards'][:]).t()
            else:
                inputs = torch.from_numpy(f['data'][:])
            targets = torch.from_numpy(f['labels'][:])
        
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        if opts.batch_size > 0: # mini-batch
            if dtype == 'train':
                batch_size = 1
                path = 'datasets/{dataset}'.format(dataset=opts.dataset)
                batch_sampler = samplers.BalancedRandomBatchSampler(
                                    dataset, batch_size=opts.batch_size,
                                    num_epochs=opts.num_epochs,
                                    balance=(opts.batch_size >= 5*targets.max()), path=path)
            else:
                batch_size = min(opts.batch_size*10, len(dataset))
                batch_sampler = None
        else: # full-batch
            batch_size = len(dataset)
            batch_sampler = None
        data_loader[dtype] = torch.utils.data.DataLoader(
                                 dataset, batch_size=batch_size, batch_sampler=batch_sampler,
                                 num_workers=opts.workers, pin_memory=True)
        
        print('{dtype} data_loader; {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))
    
    return data_loader

def relabel(rate, labels, T, num_epochs=50, path=None, start_time=time.time()):

    relabels_path = '{path}/relabels_{rate:d}.h5'.format(path=path, rate=rate)
    relabel_rate = rate / 100.
    if path is not None and os.path.isfile(relabels_path):
        with h5py.File(relabels_path, 'r') as f:
            relabels = f['relabels'][:]
    else:
        num_data = len(labels)
        num_leaves = len(T['wnids_leaf'])
        parents = T['parents']
        descendants = T['descendants']
        relabels = np.zeros([num_epochs, num_data], dtype=int)
        for epoch in range(num_epochs):
            relabels[epoch] = labels.numpy()
            visited = set()
            unvisited = list(range(num_leaves))
            while unvisited:
                k = unvisited[0]
                unvisited = unvisited[1:]
                descendants_proper_k = set(descendants[k])
                descendants_proper_k.discard(k)
                if visited.intersection(descendants_proper_k) == set(descendants_proper_k):
                    if len(parents[k]) == 0:
                        visited.add(k)
                        continue
                    num_data_k = (relabels[epoch] == k).sum()
                    relabel_me = (relabels[epoch] == k).nonzero()[0][np.random.rand(num_data_k) < relabel_rate]
                    rands = np.random.rand(len(relabel_me))
                    num_parents_k = len(parents[k])
                    for i in range(num_parents_k):
                        relabels[epoch][relabel_me[(rands >= i/num_parents_k) & \
                                                   (rands <= (i+1)/num_parents_k)]] = parents[k][i]
                        if parents[k][i] not in unvisited:
                            unvisited.append(parents[k][i])
                    visited.add(k)
                else:
                    unvisited.append(k)
                assert visited.intersection(unvisited) == set(), 'revisit ' + str(visited.intersection(unvisited))
            print_freq = min([100, (num_epochs-1) // 5 + 1])
            print_me = (epoch == 0 or epoch == num_epochs-1 or (epoch+1) % print_freq == 0)
            if print_me:
                print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch+1, num_epochs=num_epochs), end='')
                print('generate relabel {rate:2d}%; {time:8.3f} s'.format(rate=rate, time=time.time()-start_time))
        
        if path is not None:
            with h5py.File(relabels_path, 'w') as f:
                f.create_dataset('relabels', data=relabels, compression='gzip', compression_opts=9)
    
    print('relabel {rate:2d}%; {time:8.3f} s'.format(rate=rate, time=time.time()-start_time))
    return torch.from_numpy(relabels)

def relabel_batch(rate, labels, T):
    root = T['root']
    parents = T['parents']
    relabel_rate = rate / 100.
    relabels = labels.clone()
    relabel_me = (relabels != root)
    while relabel_me.sum():
        relabel_me &= (torch.rand(relabels.size(0)) < relabel_rate)
        for i in relabel_me.nonzero().view(-1):
            k = relabels[i]
            if len(parents[k]) == 0:
                relabel_me[i] = False
            elif len(parents[k]) == 1:
                relabels[i] = parents[k][0]
            else:
                relabels[i] = parents[k][int(torch.rand(1)*len(parents[k]))]
    return relabels

def update_taxonomy(method, T, radius=-1, start_time=time.time()):

    num_leaves = len(T['wnids_leaf'])
    num_classes = len(T['wnids'])
    num_supers = num_classes - num_leaves
    
    children = T['children']
    is_ancestor_mat = T['is_ancestor_mat']
    num_children = T['num_children']
    ch_slice = T['ch_slice']
    
    if ('LOO' in method) or ('RLB' in method):
        relevant = [[]]*ch_slice[-1]
        labels_relevant = np.zeros([num_classes, ch_slice[-1]], dtype=bool)
        for k in range(num_leaves, num_classes):
            m = k - num_leaves
            for i_ch, ch in enumerate(children[k]):
                non_de = ~is_ancestor_mat[ch]
                non_de[k] = False
                relevant[ch_slice[m]+i_ch] = np.concatenate([[k], non_de.nonzero()[0]], axis=0)
                labels_relevant[:, ch_slice[m]+i_ch] = ~non_de
        
        # in models.LOOLoss, filter training data with "labels_relevant", gather scores with "relevant"
        
        # relevant: for train & val
        # relevant[ch_slice[m]+i_ch] = [target, {exclusive,}]
        # (i_ch)-th child of the (m)-th super class is considered as a positive class (= target)
        # leaves under the (m)-th super class exclusive to the target are considered as negative classes
        T['relevant'] = relevant
        
        # labels_relevant: for train
        # labels_relevant[k, ch_slice[m]+i_ch] = True if the (k)-th class is under the positive class
        # len(relevant[j]) + labels_relevant[:, j].sum() == num_classes + 1
        T['labels_relevant'] = labels_relevant 
    
    elif method == 'TD':
        # In ImageNet ILSVRC 2012 1k, num_regu[-1] == num_regu[12]
        if radius >= 0:
            if 'dist_mat' not in T:
                raise AssertionError('dist_mat is missing; build detailed taxonomy')
            dist_mat = T['dist_mat']
        
        multi_inds = [[[] for _ in range(num_classes)] for _ in range(num_supers)]
        labels_ch = -np.ones([num_classes, ch_slice[-1]], dtype=int)
        labels_in = np.zeros([num_classes, num_supers], dtype=bool)
        labels_out = np.zeros([num_classes, num_supers], dtype=bool)
        classifiable = [[]]*num_supers
        num_regu = np.zeros(num_supers, dtype=int)
        for k in range(num_leaves, num_classes):
            m = k - num_leaves
            for i_ch, ch in enumerate(children[k]):
                for de in is_ancestor_mat[ch].nonzero()[0]:
                    multi_inds[m][de].append(i_ch)
                labels_ch[:, ch_slice[m]+i_ch] = i_ch*is_ancestor_mat[ch].astype(int) \
                                                 - (~is_ancestor_mat[ch]).astype(int)
            labels_in[:,m] = is_ancestor_mat[k]
            # labels_in[k,m] = False # if out regularizes k
            
            if radius == -1:
                b_regu_out = ~is_ancestor_mat[k]
                # b_regu_out[k] = True # if out regularizes k
                classifiable[m] = list(range(num_classes))
            else:
                b_regu_an = is_ancestor_mat[:,k] & (dist_mat[k] <= radius) # ancestors within (radius)
                b_regu_out = ~is_ancestor_mat[k] & is_ancestor_mat[b_regu_an].any(axis=0)
                # b_regu_out[k] = True # if out regularizes k
                classifiable[m] = (is_ancestor_mat[k] | b_regu_out).nonzero()[0].tolist()
            labels_out[:,m] = b_regu_out
            num_regu[m] = b_regu_out[:num_leaves].sum()
        
        # multi_inds, classifiable: for val
        
        # multi_inds[m][k] = [{i_ch,}] are true labels of the (k)-th class at the (m)-th super class
        T['multi_inds'] = multi_inds
        
        # classifiable[m] = [{k,}] are labels under the ancestors of the (m)-th super class within (radius)
        # classifiable[m] = range(num_classes) for all m if radius == -1
        T['classifiable'] = classifiable
        
        # labels_ch, labels_in, labels_out: for train
        
        # labels_ch[k, ch_slice[m]+i_ch] = i_ch if k is under the (i_ch)-th child of the (m)-th super class; else -1
        T['labels_ch'] = labels_ch
        
        # labels_in[k,m] = True if k is under the (m)-th super class
        T['labels_in'] = labels_in
        
        # labels_out[k,m] = True if k is not under the (m)-th super class
        #                        but under the ancestors of the (m)-th super class within (radius)
        # (labels_in + labels_out).all() = True if radius == -1
        T['labels_out'] = labels_out
        
        # num_regu: not used
        # num_regu[m] is the number of leaves in labels_out[m]
        T['num_regu'] = num_regu
    
    elif method == 'ZSL':
        root = T['root']
        label_zsl = T['label_zsl']
        
        multi_probs = np.zeros([ch_slice[-1], num_classes])
        multi_probs_class = np.zeros([num_classes, num_classes])
        multi_probs_class[root] = 1.
        for k in range(num_leaves, num_classes):
            m = k - num_leaves
            num_belong = np.sum(is_ancestor_mat[children[k]], axis=0)
            b_belong = num_belong > 0 # b_belong == is_ancestor_mat[k] except b_belong[k]
            num_belong[num_belong == 0] = 1
            multi_probs[ch_slice[m]:ch_slice[m+1]] = multi_probs_class[children[k]] = \
                b_belong * is_ancestor_mat[children[k]] / num_belong[None, :] + ~b_belong / num_children[m]
        multi_probs = multi_probs.T
        multi_probs_class = multi_probs_class.T
        
        # ideal output probabilities; see Appendix D.1
        T['multi_probs'] = multi_probs
        T['multi_probs_class'] = multi_probs_class
        T['att'] = multi_probs[label_zsl, :] # for DAG
        T['attr'] = multi_probs_class[label_zsl, :] # for tree
    else:
        print('no taxonomy update; unidentifiable method: {method}'.format(method=method))
    
    print('taxonomy for {method}; {time:8.3f} s'.format(method=method, time=time.time()-start_time))

def get_path(opts):

    path = 'train/{dataset}/{cnn}/{method}/' \
                .format(dataset=opts.dataset, cnn=opts.cnn, method=opts.method)
    if 'TD+' in opts.method:
        path += '{td_name}/'.format(td_name=opts.td_name)
    if not os.path.isdir(path):
            os.makedirs(path)
    path += '{method}_'.format(method=opts.method)
    if opts.method == 'TD':
        path += '{param:d}_'.format(param=opts.radius)
        path += '{param:.0e}_'.format(param=opts.ex_smooth)
    elif 'RLB' in opts.method:
        path += '{param:d}_'.format(param=opts.relabel)
    elif 'LOO' in opts.method:
        path += '{param:.0e}_'.format(param=opts.loo)
    path += '{param:.0e}_'.format(param=opts.label_smooth)
    if opts.num_layers > 0:
        if opts.no_last_relu:
            path += 'l{num_layers:d}n_'.format(num_layers=opts.num_layers)
        else:
            path += 'l{num_layers:d}_'.format(num_layers=opts.num_layers)
    if opts.novel_score and opts.method == 'TD':
        path += 'ns_'
    if opts.class_wise and opts.method in ['TD', 'LOO', 'TD+LOO']:
        path += 'cw_'
    if opts.test_relu and ('TD+' in opts.method):
        path += 'relu_'
    if 'TD+' in opts.method:
        path += '{param}_'.format(param=opts.softmax)
    path += '{wd:.0e}_{lr:.0e}'.format(wd=opts.wd, lr=opts.lr)
    return path

def load_model(model, optimizer, scheduler, path, num_epochs, start_time=time.time()):

    epoch = num_epochs
    while epoch > 0 and not os.path.isfile('{path}_model_{epoch:d}.pth'.format(path=path, epoch=epoch)):
        epoch -= 1
    if epoch > 0:
        model_path = '{path}_model_{epoch:d}.pth'.format(path=path, epoch=epoch)
        model_state_dict = torch.load('{path}_model_{epoch:d}.pth'.format(path=path, epoch=epoch))
        model.load_state_dict(model_state_dict)
        if optimizer is not None:
            optimizer_state_dict = torch.load('{path}_optimizer_{epoch:d}.pth'.format(path=path, epoch=epoch))
            optimizer.load_state_dict(optimizer_state_dict)
        if scheduler is not None:
            scheduler_state_dict = torch.load('{path}_scheduler_{epoch:d}.pth'.format(path=path, epoch=epoch))
            scheduler.best = scheduler_state_dict['best']
            scheduler.cooldown_counter = scheduler_state_dict['cooldown_counter']
            scheduler.num_bad_epochs = scheduler_state_dict['num_bad_epochs']
            scheduler.last_epoch = scheduler_state_dict['last_epoch']
        print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch, num_epochs=num_epochs), end='')
        print('load {path}; '.format(path=model_path), end='')
        print('{time:8.3f} s'.format(time=time.time()-start_time))
    return epoch

def save_model(model, optimizer, scheduler, path, epoch, num_epochs, prev_epoch=0, start_time=time.time()):

    model_state_dict = copy.deepcopy(model).cpu().state_dict()
    model_path = '{path}_model_{epoch:d}.pth'.format(path=path, epoch=epoch)
    torch.save(model_state_dict, model_path)
    if optimizer is not None:
        optimizer_state_dict = optimizer.state_dict()
        optimizer_path = '{path}_optimizer_{epoch:d}.pth'.format(path=path, epoch=epoch)
        torch.save(optimizer_state_dict, optimizer_path)
    if scheduler is not None:
        scheduler_state_dict = {'best': scheduler.best,
                                'cooldown_counter': scheduler.cooldown_counter,
                                'num_bad_epochs': scheduler.num_bad_epochs,
                                'last_epoch': scheduler.last_epoch}
        scheduler_path = '{path}_scheduler_{epoch:d}.pth'.format(path=path, epoch=epoch)
        torch.save(scheduler_state_dict, scheduler_path)
    print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch, num_epochs=num_epochs), end='')
    print('save {path}; '.format(path=model_path), end='')
    print('{time:8.3f} s'.format(time=time.time()-start_time))
    
    # remove previous model
    if prev_epoch > 0:
        prev_model_path = '{path}_model_{epoch:d}.pth'.format(path=path, epoch=prev_epoch)
        if os.path.isfile(prev_model_path):
            os.remove(prev_model_path)
        prev_optimizer_path = '{path}_optimizer_{epoch:d}.pth'.format(path=path, epoch=prev_epoch)
        if os.path.isfile(prev_optimizer_path):
            os.remove(prev_optimizer_path)
        prev_scheduler_path = '{path}_scheduler_{epoch:d}.pth'.format(path=path, epoch=prev_epoch)
        if os.path.isfile(prev_scheduler_path):
            os.remove(prev_scheduler_path)
        print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch, num_epochs=num_epochs), end='')
        print('remove {path}; '.format(path=prev_model_path), end='')
        print('{time:8.3f} s'.format(time=time.time()-start_time))

if __name__ == '__main__':
    opts = argparse.Namespace()
    opts.gpu = True
    opts.workers = 8
    opts.dataset = 'ImageNet'
    opts.cnn = 'resnet101'
    opts.batch_size = 256
    print(opts)
    
    start_time = time.time()
    
    dtypes = ['train', 'val', 'known', 'novel']
    extract_features(dtypes, opts, start_time)
