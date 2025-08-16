import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchvision.transforms import transforms

import optuna

from alexnet import AlexNet
from imagenet import load_imagenet1k_mini

EPOCHS = 5
VERBOSE = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')


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


def train(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module
    ) -> None:
    model.train()

    for epoch in range(EPOCHS): 
        for input, target in dataloader:
            input = input.to(DEVICE)
            target = target.to(DEVICE) 
            
            pred = model(input)
            loss = criterion(pred, target)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
        if VERBOSE: 
            acc = evaluate(model, dataloader, criterion)
            print(f'epoch: {epoch}\taccuracy: {acc}\terror: {1 - acc}')