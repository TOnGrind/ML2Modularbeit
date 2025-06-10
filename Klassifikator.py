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
from torch.utils.data import DataLoader


def configuartion(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
num_epochs = 30,
batch_size = 64,
initial_lr = 0.1,
class_list = list('0123456789ABCDEFGHIJKLMabcdefghijklm')):
    # -----------------------------
    # Geräte-Konfiguration , Hyperparameter ,Klassenliste
    # -----------------------------  

    return device, num_epochs, batch_size, initial_lr ,class_list



# -----------------------------
# Optuna-Ziel-Funktion
# -----------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
    return total_loss / len(data_loader)


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

    
def get_objective(train_dataset, test_dataset, device, model, early_stopping):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 144, 256])
        lr = trial.suggest_float("lr", 1e-4, 0.1, log=True)
        momentum = trial.suggest_float("momentum", 0.6, 0.95)
        step_size = trial.suggest_int("step_size", 2, 5)
        gamma = trial.suggest_float("gamma", 0.5, 0.95)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        

        for epoch in range(20):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate_model(model, test_loader, criterion, device)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"⛔ Early Stopping in Epoch {epoch+1}")
                break

            print(f"📉 Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
            scheduler.step()

        accuracy = compute_accuracy(model, test_loader, device)
        return accuracy

    return objective



# -----------------------------
# Early Stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True





# -----------------------------
# CNN-Modell
# -----------------------------


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class BasicBlock2(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, negative_slope=0.01):
        super(BasicBlock2, self).__init__()
        
        # Erste Convolutional Layer mit BatchNorm und Leaky ReLU
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Zweite Convolutional Layer mit BatchNorm und Leaky ReLU
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut-Block (Falls notwendig)
        self.shortcut = nn.Identity()  # Verwenden von nn.Identity() für den Fall ohne Transformation
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
        # Negative slope für LeakyReLU
        self.negative_slope = negative_slope

    def forward(self, x):
        # Erste Convolution und LeakyReLU Aktivierung
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.negative_slope)
        
        # Zweite Convolution und BatchNorm
        out = self.bn2(self.conv2(out))
        
        # Shortcut Verknüpfung hinzufügen
        out += self.shortcut(x)
        
        # Letzte Aktivierung
        return F.leaky_relu(out, negative_slope=self.negative_slope)




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Eingabe-Channel von 3 auf 1 ändern
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # Hier 1 statt 3
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def ResNet18(BasicBlock = BasicBlock2,num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Hilfsfunktion zur Berechnung der ROC-AUC für das Multiklassenproblem
def model_output_to_probs(model, test_loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


