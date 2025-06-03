import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets
from torchvision.transforms import ToTensor, RandomAffine, Compose
from PIL import Image
from collections import defaultdict


def configuartion(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
num_epochs = 20,
batch_size = 64,
initial_lr = 0.2,
class_list = list('0123456789ABCDEFGHIJKLMabcdefghijklm')):
    # -----------------------------
    # Geräte-Konfiguration , Hyperparameter ,Klassenliste
    # -----------------------------  

    return device, num_epochs, batch_size, initial_lr ,class_list


# -----------------------------
# CNN-Modell
# -----------------------------


class ConvNet(nn.Module):
    def __init__(self,class_list):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(class_list))  # 36 Klassen

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


class EMNISTNetPlus(nn.Module):
    def __init__(self, num_classes):
        super(EMNISTNetPlus, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x
