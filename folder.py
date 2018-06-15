import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def find_classes(dataset, dtype):
    dtype_ = dtype if dtype == 'novel' else 'known'
    path = 'taxonomy/{dataset}/classes_{dtype}.txt'.format(dataset=dataset, dtype=dtype_)
    if os.path.isfile(path):
        classes = open(path, 'r').read().strip().replace('\t', '\n').splitlines()[::2]
    else:
        raise FileNotFoundError('{path} missing'.format(path=path))
    return classes

def make_dataset(dataset, dtype):
    path = 'taxonomy/{dataset}/images_{dtype}.txt'.format(dataset=dataset, dtype=dtype)
    if os.path.isfile(path):
        images_list = open(path, 'r').read().strip().replace('\t', '\n').splitlines()
        images = list(zip(images_list[0::2], [int(c) for c in images_list[1::2]]))
    else:
        raise FileNotFoundError('{path} missing'.format(path=path))
    return images


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


class StaticImageFolder(data.Dataset):

    def __init__(self, dataset='ImageNet', dtype='novel', transform=None, target_transform=None,
                 loader=default_loader):
        classes = find_classes(dataset, dtype)
        imgs = make_dataset(dataset, dtype)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = None
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = None
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
