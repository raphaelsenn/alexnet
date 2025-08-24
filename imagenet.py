import os

import torch
from torch.utils.data import Subset

from torchvision import datasets
from torchvision.datasets import ImageNet

def load_imagenet1k_mini(
        root_dir: str='./imagenet-mini/',
        transform=None,
        val_size: float | None = None, 
    ) -> tuple:

    dataset = datasets.ImageFolder(
        os.path.join(root_dir, 'train'), 
        transform
    )    
    test_set = datasets.ImageFolder(
        os.path.join(root_dir, 'val'),
        transform
    )    
    
    if val_size:
        assert 0 < val_size < 1, 'Validation size needs to be between 0 and 1'

        train_size = int((1 - val_size) * len(dataset))
        idx = torch.randperm(len(dataset))

        train_idx = idx[:train_size]
        val_idx = idx[train_size:]

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        return train_set, val_set, test_set

    return dataset, test_set


def load_imagenet1k(
        root_dir: str='./imagenet/',
        transform=None,
        eval_transform=None,
        val_size: float | None = None,
        seed: int = 42
    ) -> tuple:

    dataset_train = ImageNet(root_dir, split='train', transform=transform)
    dataset_train_eval = ImageNet(root_dir, split='train', transform=eval_transform)
    dataset_test = ImageNet(root_dir, split='val', transform=eval_transform)

    if val_size is None:
        return dataset_train, dataset_test

    assert 0 < val_size < 1, 'Validation size needs to be between 0 and 1'
    N_train = len(dataset_train) 
    train_size = int((1.0 - val_size) * N_train)

    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(dataset_train), generator=g)

    train_idx = idx[:train_size]
    val_idx = idx[train_size:]

    train_set = Subset(dataset_train, train_idx)
    val_set = Subset(dataset_train, val_idx)
    return train_set, val_set, dataset_test