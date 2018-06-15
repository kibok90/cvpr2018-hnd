import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler

class BalancedRandomBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, num_epochs=50, balance=True, path=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.order = balanced_shuffle(data_source.target_tensor, num_epochs, path) \
                     if balance else shuffle(data_source.target_tensor, num_epochs, path)
        self.epoch = 0
    
    def __iter__(self):
        for i in range(len(self)):
            batch = self.order[self.epoch, i*self.batch_size:(i+1)*self.batch_size]
            batch = batch[batch >= 0].tolist()
            yield iter(batch)
        self.epoch += 1
        if self.epoch >= self.order.size(0):
            self.epoch = 0
    
    def __len__(self):
        return self.order.size(1) // self.batch_size

def balanced_shuffle(labels, num_epochs=50, path=None, start_time=time.time()):

    order_path = '{path}/balanced_order_{num_epochs}.h5' \
                       .format(path=path, num_epochs=num_epochs)
    if path is not None and os.path.isfile(order_path):
        with h5py.File(order_path, 'r') as f:
            order = f['order'][:]
    else:
        evenness = 5 # batch_size | evenness*num_classes
        classes = np.unique(labels.numpy())
        num_classes = len(classes)
        loc_data_per_class = [np.argwhere(labels.numpy() == k).flatten() for k in classes]
        num_data_per_class = [(labels.numpy() == k).sum() for k in classes]
        max_data_per_class = max(num_data_per_class)
        num_loc_split = (max_data_per_class // evenness) * np.ones(evenness, dtype=int)
        num_loc_split[:(max_data_per_class % evenness)] += 1
        loc_split = [0]
        loc_split.extend(np.cumsum(num_loc_split).tolist())
        order = -np.ones([num_epochs, max_data_per_class*num_classes], dtype=int)
        for epoch in range(num_epochs):
            order_e = -np.ones([max_data_per_class, num_classes], dtype=int)
            for k in classes:
                loc_k = np.random.permutation(loc_data_per_class[k])
                for i in range(evenness):
                    loc_i = loc_k[loc_split[i]:loc_split[i+1]]
                    order_e[i:(len(loc_i)*evenness+i):evenness, k] = loc_i
            order[epoch] = order_e.flatten()
            print_freq = min([100, (num_epochs-1) // 5 + 1])
            print_me = (epoch == 0 or epoch == num_epochs-1 or (epoch+1) % print_freq == 0)
            if print_me:
                print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch+1, num_epochs=num_epochs), end='')
                print('generate balanced random order; {time:8.3f} s'.format(time=time.time()-start_time))
        
        if path is not None:
            with h5py.File(order_path, 'w') as f:
                f.create_dataset('order', data=order, compression='gzip', compression_opts=9)
    
    print('balanced random order; {time:8.3f} s'.format(time=time.time()-start_time))
    return torch.from_numpy(order)

def shuffle(labels, num_epochs=50, path=None, start_time=time.time()):

    order_path = '{path}/order_{num_epochs}.h5' \
                       .format(path=path, num_epochs=num_epochs)
    if path is not None and os.path.isfile(order_path):
        with h5py.File(order_path, 'r') as f:
            order = f['order'][:]
    else:
        order = -np.ones([num_epochs, labels.size(0)], dtype=int)
        for epoch in range(num_epochs):
            order[epoch] = np.random.permutation(labels.size(0))
            print_freq = min([100, (num_epochs-1) // 5 + 1])
            print_me = (epoch == 0 or epoch == num_epochs-1 or (epoch+1) % print_freq == 0)
            if print_me:
                print('{epoch:4d}/{num_epochs:4d} e; '.format(epoch=epoch+1, num_epochs=num_epochs), end='')
                print('generate random order; {time:8.3f} s'.format(time=time.time()-start_time))
        
        if path is not None:
            with h5py.File(order_path, 'w') as f:
                f.create_dataset('order', data=order, compression='gzip', compression_opts=9)
    
    print('random order; {time:8.3f} s'.format(time=time.time()-start_time))
    return torch.from_numpy(order)
