# Introduction
This repository implements [Lee et al., Hierarchical Novelty Detection for Visual Object Recognition, CVPR 2018](https://arxiv.org/abs/1804.00722) in PyTorch.
```
@inproceedings{lee2018hierarchical,
  title={Hierarchical Novelty Detection for Visual Object Recognition},
  author={Lee, Kibok and Lee, Kimin and Min, Kyle and Zhang, Yuting and Shin, Jinwoo and Lee, Honglak},
  booktitle={CVPR},
  year={2018}
}
```

# Dependencies
- [Python 3.6.2](https://www.python.org/downloads/) (or https://www.anaconda.com/download/)
- [Pytorch 0.3.0.post4](https://pytorch.org/) (since Pytorch frequently updates their libraries, our code would not work if you have a different version.)
- [Torchvision 0.2.0](https://github.com/pytorch/vision)
- [NumPy 1.14.0](https://pypi.org/project/numpy)
- [SciPy 0.19.1](https://pypi.org/project/scipy) (to load features in mat format for AWA and CUB experiments)
- [h5py 2.7.0](https://pypi.org/project/h5py/) (to save features and random numbers)
- [matplotlib 2.0.2](https://pypi.org/project/matplotlib/) (to plot known-novel class accuracy curve)
- [NLTK 3.2.4](https://pypi.org/project/nltk/) (to convert WordNet synset to offset ids for CUB experiments)

# Data
You may download either raw images or ResNet-101 features. If you download ResNet-101 features, place them in `datasets/{dataset}/`. (`{dataset} = ImageNet, AWA2, CUB`)

### ImageNet
- [Raw images](http://image-net.org/download)
  - Move ILSVRC 2012 train to `datasets/ImageNet/train/`.
    - e.g., an image should be found in `datasets/ImageNet/train/n01440764/n01440764_18.JPEG`.
  - Move ILSVRC 2012 val to `datasets/ImageNet/known/`.
    - ILSVRC 2012 validation dataset is not sorted. You can move validation images to labeled subfolders using [[this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)].
  - Move ImageNet Fall 2011 to `datasets/ImageNet/novel/`.
- ResNet-101 features [[train](https://drive.google.com/uc?export=download&id=1sFrzU_2W8aZUKgnzxI4lAVHqDupIN9Lv)] [[val](https://drive.google.com/uc?export=download&id=1r6--AVTY2_Na3CF9vfzaXQ5MSu-aYG8l)] [[known](https://drive.google.com/uc?export=download&id=1GXdD9eOIse6YPGWFY7LFCKB6uj12Ob99)] [[novel](https://drive.google.com/uc?export=download&id=17F34X0r_wOfvbcNHWzs1JGtLCkcyFmCU)]

### AWA, CUB
- [ResNet-101 features](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip)

### WordNet
You do not have to download the files, but we provide the source of them for your reference.
- [WordNet](http://www.image-net.org/archive/wordnet.txt)
- [WordNet is-a relationship](http://www.image-net.org/archive/wordnet.is_a.txt)

# Preparation

### Taxonomy
Run `sh scripts/preparation.sh {d}`. (`{d} = imagenet_full, imagenet, awa2, cub`)
- `{d} = imagenet_full` if you have raw images
- `{d} = imagenet` if you have ResNet-101 features

Output files are in `taxonomy/{dataset}/`.

You can download pre-built taxonomies [[here](https://drive.google.com/uc?export=download&id=1p_-Yq3hd9ITrAveafj3sC00VCN3VJlE1)].

### Feature extraction (ImageNet) or conversion (AWA, CUB)
Run `sh scripts/feature.sh {d}`. (`{d} = imagenet, awa2, cub`)

Output files are in `datasets/{dataset}/`.

If you have ResNet-101 features for ImageNet, skip this.

# Train, test
Run `sh scripts/train.sh {d} {m}`. (`{d} = imagenet, awa2, cub, {m} = relabel, td, loo, td+loo`)

Output files are in `train/`.

You can download pre-trained models [[here](https://drive.google.com/uc?export=download&id=1y1CZAJZiVabFaTiim8sr89j_KqDs3bFv)].

# Note
- The code keeps all random numbers and final models. For new experiment, you may remove the following if exist:
```
datasets/{dataset}/balanced_order_{:d}.h5
datasets/{dataset}/relabels_{:d}.h5
train/
```
- The code can produce results in hierarchical measures. To see them, build the taxonomy with additional argument, e.g., `python build_taxonomy.py ImageNet detailed`
