import os

from torch.utils.data import Dataset

from torchvision import datasets
import torchvision
import torchvision.transforms as TF


def load_imagenet1k_mini(
        root_dir: str='./imagenet-mini/',
        train_val_test: bool=False,
        transform=None
    ) -> tuple:

    train_set = datasets.ImageFolder(
        os.path.join(root_dir, 'train'), 
        transform
    )    
    test_set = datasets.ImageFolder(
        os.path.join(root_dir, 'val'),
        transform
    )    

    return train_set, test_set