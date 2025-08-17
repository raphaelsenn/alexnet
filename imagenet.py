import os

import torch
from torch.utils.data import Subset

from torchvision import datasets


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