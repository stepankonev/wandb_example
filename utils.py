import torch.optim as optim
import torch.nn as nn
import yaml
import argparse
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, help='Path to the config file')
    args = parser.parse_args()
    return args


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_optimizer(name, model, parameters):
    if name == 'SGD':
        return optim.SGD(model.parameters(), **parameters)
    elif name == 'Adam':
        return optim.Adam(model.parameters(), **parameters)
    elif name == 'RMSprop':
        return optim.RMSprop(model.parameters(), **parameters)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


# A model with a few linear layers and ReLU activations can be created using the following function:
def get_model(n_layers, add_batchnorm):
    assert n_layers > 0
    if n_layers == 1:
        return nn.Sequential(
            nn.Linear(784, 10)
        )
    layers = [nn.Linear(784, 256), nn.ReLU()]
    if add_batchnorm:
        layers.append(nn.BatchNorm1d(256))
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(256, 256), nn.ReLU()])
        if add_batchnorm:
            layers.append(nn.BatchNorm1d(256))
    layers.append(nn.Linear(256, 10))
    return nn.Sequential(*layers)


# Get transform for mnist dataset that flattens and returns a tensor
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])