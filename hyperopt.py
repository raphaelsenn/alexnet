import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchvision.transforms import transforms

import optuna

from alexnet import AlexNet
from imagenet import load_imagenet1k_mini


# Hyperparameters and Settings
EPOCHS = 2
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_CUDA = DEVICE.type == 'cuda'

torch.manual_seed(SEED)
if TORCH_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(SEED)

# Transformations
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module
    ) -> None:
    model.train()
    for input, target in dataloader:
        input = input.to(DEVICE)
        target = target.to(DEVICE) 
        
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> float:
    model.eval() 
    correct = 0
    total = 0

    for input, target in dataloader:
        input = input.to(DEVICE)
        target = target.to(DEVICE) 
        pred = model(input)
        pred = torch.argmax(pred, dim=1)        

        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / total


def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1)
    momentum = trial.suggest_float('momentum', 0.8, 0.95)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)

    model = AlexNet().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_set, val_set, _ = load_imagenet1k_mini(transform=transform, val_size=0.1)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    for _ in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion)
    
    return evaluate(model, val_loader, criterion)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)