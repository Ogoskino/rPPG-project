import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define RGB and Thermal models here (RGBModel and ThermalModel classes)

# Dataset definitions
class Dataset(Dataset):
    def __init__(self, data, labels):
        self.rgb_data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.rgb_data[idx]
        label = self.labels[idx]

        return image, label