from PIL import Image
import os
import os.path

import sys
import numpy as np
import multiprocessing as mp
import time

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(id, dataset, dtype, filename):
    filename_lower = filename.lower()
    if any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS):
        if dtype == 'novel':
            try:
                default_loader(filename)
                return True
            except OSError:
                print('{filename} failed to load'.format(filename=filename))
                with open('taxonomy/{dataset}/corrupted_{dtype}_{id:d}.txt' \
                          .format(dataset=dataset, dtype=dtype, id=id), 'a') as f:
                    f.write(filename + '\n')
                return False
        else:
            return True
    else:
        return False

def find_classes(id, num_workers, dataset, dtype):
    dir = 'datasets/{dataset}/{dtype}'.format(dataset=dataset, dtype=dtype)
    classes_path = 'taxonomy/{dataset}/classes_{dtype}_{id:d}.txt'.format(dataset=dataset, dtype=dtype, id=id)
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    num_classes = len(classes)
    with open(classes_path, 'w') as f:
        for cname in classes[id*num_classes//num_workers:(id+1)*num_classes//num_workers]:
            num = len(os.listdir(os.path.join(dir, cname)))
            f.write('{cname}\t{num}\n'.format(cname=cname, num=num))
    return classes


def make_dataset(id, num_workers, dataset, dtype, classes, bias, max_num_images):
    dir = 'datasets/{dataset}/{dtype}'.format(dataset=dataset, dtype=dtype)
    if dtype == 'train':
        train_path = 'taxonomy/{dataset}/images_{dtype}_{id:d}.txt'.format(dataset=dataset, dtype='train', id=id)
        val_path   = 'taxonomy/{dataset}/images_{dtype}_{id:d}.txt'.format(dataset=dataset, dtype='val',   id=id)
        fs = [open(train_path, 'w'), open(val_path, 'w')]
    else:
        images_path = 'taxonomy/{dataset}/images_{dtype}_{id:d}.txt'.format(dataset=dataset, dtype=dtype, id=id)
        f = open(images_path, 'w')
    num_classes = len(classes)
    classes_id = list(enumerate(classes))
    for c, cname in classes_id[id*num_classes//num_workers:(id+1)*num_classes//num_workers]:
        d = os.path.join(dir, cname)
        num_images = 0
        stop_flag = False
        if dtype == 'train': f = fs[1]
        for fname in sorted(os.listdir(d)):
            path = os.path.join(d, fname)
            if is_image_file(id, dataset, dtype, path):
                num_images += 1
                f.write('{path}\t{c:d}\n'.format(path=path, c=c+bias))
                if max_num_images >= 0 and num_images >= max_num_images:
                    if dtype == 'train': f = fs[0]
                    else:
                        stop_flag = True
                        break
            if stop_flag:
                break
    if dtype == 'train':
        fs[0].close()
        fs[1].close()
    else:
        f.close()

def merge_text(num_workers, dataset, dtype, ttype):
    path = 'taxonomy/{dataset}/{ttype}_{dtype}'.format(dataset=dataset, ttype=ttype, dtype=dtype)
    with open('{path}.txt'.format(path=path), 'w') as fo:
        for id in range(num_workers):
            path_id = '{path}_{id:d}.txt'.format(path=path, id=id)
            if os.path.isfile(path_id):
                with open(path_id, 'r') as fi:
                    fo.write(fi.read())
    for id in range(num_workers):
        path_id = '{path}_{id:d}.txt'.format(path=path, id=id)
        if os.path.isfile(path_id):
            os.remove(path_id)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

if __name__ == '__main__':
    dataset = 'ImageNet'
    max_num_images = 50
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 4
    start_time = time.time()
    pool = mp.Pool(processes=num_workers)
    
    taxonomy_folder = 'taxonomy/{dataset}'.format(dataset=dataset)
    if not os.path.isdir(taxonomy_folder):
        os.makedirs(taxonomy_folder)
    
    # find classes
    for dtype in ['known', 'novel']:
        args = [(id, num_workers, dataset, dtype) for id in range(num_workers)]
        pool.starmap(find_classes, args)
        merge_text(num_workers, dataset, dtype, 'classes')
        print('{dtype} classes {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))
    
    # filter classes
    taxonomy_path = '{folder}/taxonomy.npy'.format(folder=taxonomy_folder, dataset=dataset)
    if not os.path.isfile(taxonomy_path):
        os.system('python build_taxonomy.py')
    T = np.load(taxonomy_path).item()
    
    # find images; val is extracted from train
    for dtype in ['train', 'known', 'novel']:
        classes = T['wnids_novel'] if dtype == 'novel' else T['wnids_leaf']
        bias = len(T['wnids_leaf']) if dtype == 'novel' else 0
        args = [(id, num_workers, dataset, dtype, classes, bias, max_num_images) for id in range(num_workers)]
        pool.starmap(make_dataset, args)
        merge_text(num_workers, dataset, dtype, 'images')
        if dtype == 'train': merge_text(num_workers, dataset, 'val', 'images')
        if dtype == 'novel': merge_text(num_workers, dataset, 'novel', 'corrupted')
        print('{dtype} images {time:8.3f} s'.format(dtype=dtype, time=time.time()-start_time))
