import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchvision.transforms import transforms

import optuna

from alexnet import AlexNet
from imagenet import load_imagenet1k_mini


@torch.inference_mode()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
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


if __name__ == '__main__':
    # Hyperparameters/Settings/Initialization 
    EPOCHS = 5
    BATCH_SIZE = 128
    HYPERS = {
        'lr': 0.09619116566867329, 
        'momentum': 0.8074443406469723, 
        'weight_decay': 0.00019837055273061845
    }
    SEED = 42

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    TORCH_CUDA = DEVICE.type == 'cuda'
    VERBOSE = True

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

    # Prepare model 
    model = AlexNet().to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        HYPERS['lr'], 
        HYPERS['momentum'], 
        weight_decay=HYPERS['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    # Prepare data
    train_set, test_set = load_imagenet1k_mini(transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Start training
    train(model, train_loader, optimizer, criterion)