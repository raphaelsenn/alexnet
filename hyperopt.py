import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchvision.transforms import transforms

import optuna

from alexnet import AlexNet
from imagenet import load_imagenet1k_mini

EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')

transform = transforms.Compose([
    transforms.Resize((227, 227)),  # match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
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


@torch.no_grad()
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
    lr = trial.suggest_float('lr', 1e-6, 1e-1)
    beta1 = trial.suggest_float('beta1', 0.85, 0.95)
    beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2)

    model = AlexNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr, (beta1, beta2), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_set, val_set = load_imagenet1k_mini(transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

    for _ in range(EPOCHS):
        train_epoch(model, val_loader, optimizer, criterion)
    
    return evaluate(model, val_loader, criterion)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)