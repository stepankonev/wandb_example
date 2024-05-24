import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from utils import (
    parse_args, load_config, get_optimizer, get_model, get_transform)
from tqdm import tqdm
from aggregator import Average
import wandb

def main():
    args = parse_args()
    config = load_config(args.config_path)
    wandb_id = wandb.util.generate_id()
    config_name = args.config_path.split('/')[-1].split('.')[0]
    wandb.init(
        id=wandb_id,
        project='mnist',
        entity='sdrnr',
        config=config,
        name=config_name)
    model = get_model(**config['model'])
    optimizer = get_optimizer(**config['train']['optimizer'], model=model)
    criterion = nn.CrossEntropyLoss()
    transform = get_transform()
    train_dataloader = DataLoader(
        MNIST(root='./data', train=True, download=True, transform=transform),
        **config['train']['dataloader'])
    test_dataloader = DataLoader(
        MNIST(root='./data', train=False, download=True, transform=transform),
        **config['train']['dataloader'])
    train_loss_avg = Average('train', 'loss')
    test_loss_avg = Average('test', 'loss')
    train_accuracy_avg = Average('train', 'accuracy')
    test_accuracy_avg = Average('test', 'accuracy')

    for epoch in tqdm(range(config['train']['n_epochs'])):
        model.train()
        for images, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            output = model(images.view(images.size(0), -1))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_avg.add(loss.item())
            train_accuracy_avg.add(
                (output.argmax(dim=1) == labels).float().mean().item())
        model.eval()
        with torch.no_grad():
            for images, labels in test_dataloader:
                output = model(images.view(images.size(0), -1))
                loss = criterion(output, labels)
                test_loss_avg.add(loss.item())
                test_accuracy_avg.add(
                    (output.argmax(dim=1) == labels).float().mean().item())
        metrics = {}
        metrics.update(train_loss_avg.get_dict(histogram=True))
        metrics.update(test_loss_avg.get_dict(histogram=True))
        metrics.update(train_accuracy_avg.get_dict())
        metrics.update(test_accuracy_avg.get_dict())

        try:
            os.makedirs(f'./models/{config_name}')
        except FileExistsError:
            pass
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'wandb_id': wandb_id
            },
            f'./models/{config_name}/model.pth')
        wandb.log(metrics)


if __name__ == '__main__':
    main()