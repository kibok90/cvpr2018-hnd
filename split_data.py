import sys
import time

import scipy.io as sio
import h5py
import numpy as np

start_time = time.time()
## supported datasets 
# AWA, CUB
supported_datasets = ['AWA1', 'AWA2', 'CUB']
if len(sys.argv) > 1:
    assert sys.argv[1] in supported_datasets, 'supported datasets: {s}'.format(s=str(supported_datasets))
    dataset = sys.argv[1]
else:
    dataset = 'CUB'
print('{time:8.3f} s; dataset: {dataset}'.format(time=time.time()-start_time, dataset=dataset))

cnn = 'resnet101'
i_fold, n_fold = -1, 5

dtypes = ['train', 'known', 'novel']

data_path = 'datasets/{dataset}/res101.mat'.format(dataset=dataset)        
data_dict = sio.loadmat(data_path, squeeze_me=True)
splits_path = 'datasets/{dataset}/att_splits.mat'.format(dataset=dataset)
locs_dict = sio.loadmat(splits_path, squeeze_me=True)
dtype_to_locs = {'train': 'trainval_loc', 'known': 'test_seen_loc', 'novel': 'test_unseen_loc'}

for dtype in dtypes:
    save_path = 'datasets/{dataset}/{cnn}_{dtype}.h5'.format(dataset=dataset, cnn=cnn, dtype=dtype)
    
    locs = locs_dict[dtype_to_locs[dtype]].astype(int)-1
    data = data_dict['features'].astype(np.float32).T[locs]
    labels = data_dict['labels'].astype(int)[locs]-1
    
    T = np.load('taxonomy/{dataset}/taxonomy.npy'.format(dataset=dataset)).item()
    labels = np.array(T['label_enum'], dtype=int)[labels]
    
    if dtype == 'train':
        val_path = 'datasets/{dataset}/{cnn}_{dtype}.h5'.format(dataset=dataset, cnn=cnn, dtype='val')
        if i_fold >= 0: # split train val
            b_val = np.zeros_like(labels, dtype=bool)
            for l in np.unique(labels):
                l_locs = (labels == l).nonzero()[0]
                b_val[l_locs[(l_locs.shape[0]*i_fold//n_fold):(l_locs.shape[0]*(i_fold+1)//n_fold)]] = True
            
            # save
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('data',   data=data[~b_val],   compression='gzip', compression_opts=9)
                f.create_dataset('labels', data=labels[~b_val], compression='gzip', compression_opts=9)
            with h5py.File(val_path, 'w') as f:
                f.create_dataset('data',   data=data[b_val],    compression='gzip', compression_opts=9)
                f.create_dataset('labels', data=labels[b_val],  compression='gzip', compression_opts=9)
        else: # train == val
            # save
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('data',   data=data,   compression='gzip', compression_opts=9)
                f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
            with h5py.File(val_path, 'w') as f:
                f.create_dataset('data',   data=data,   compression='gzip', compression_opts=9)
                f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
    else:
        # save
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('data',   data=data,   compression='gzip', compression_opts=9)
            f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
    
    print('{dtype}; {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))
