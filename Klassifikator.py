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
import os
import optuna


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

    
def get_objective(train_dataset, test_dataset,  device, model,  early_stopping):
    def objective(trial):
        # Modell initialisieren
       
        
        # Hyperparameter vorschlagen
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 144, 256])
        lr = trial.suggest_float("lr", 1e-4, 0.1, log=True)
        momentum = trial.suggest_float("momentum", 0.6, 0.95)
        step_size = trial.suggest_int("step_size", 2, 5)
        gamma = trial.suggest_float("gamma", 0.5, 0.95)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Optimierungskomponenten
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        # Early Stopping ohne Dateisystem
       

        best_accuracy = 0
        should_prune = False

        try:
            for epoch in range(20):
                # Training
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                
                # Evaluation
                val_loss = evaluate_model(model, test_loader, criterion, device)
                accuracy = compute_accuracy(model, test_loader, device)
                
                # Scheduler und Early Stopping
                scheduler.step()
                early_stopping(val_loss, model)
                
                # Fortschritt melden
                trial.report(accuracy, epoch)
                
                # Prüfen auf Pruning
                if trial.should_prune():
                    should_prune = True
                    break
                
                if early_stopping.early_stop:
                    print(f"⛔ Early Stopping in Epoch {epoch+1}")
                    break
                
                # Beste Accuracy aktualisieren
                if accuracy > best_accuracy:
                    best_accuracy = accuracy

        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()
        
        # Besten Modellzustand wiederherstellen (falls vorhanden)
        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)
            final_accuracy = compute_accuracy(model, test_loader, device)
        else:
            final_accuracy = best_accuracy
        
        # Bei Pruning die beste bisherige Accuracy zurückgeben
        if should_prune:
            raise optuna.exceptions.TrialPruned()
        
        return final_accuracy

    return objective



# -----------------------------
# Early Stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_state = None  # Speichert den Modellzustand im RAM

    def __call__(self, val_loss, model):
        score = -val_loss  # Höhere Werte = besser
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Speichert Modellzustand im RAM"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model state.')
        
        # Modellzustand im RAM speichern (keine Dateioperation)
        self.best_model_state = model.state_dict().copy()
        self.val_loss_min = val_loss


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


class TypeClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)  # 3 Typen
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ModularClassifier(nn.Module):
    def __init__(self, tm1, tm2, class_type_map):
        super().__init__()
        self.tm1 = tm1
        self.tm2 = tm2
        self.register_buffer('class_type_map', torch.tensor(class_type_map, dtype=torch.long))
    
    def forward(self, x):
        # Logits zu Wahrscheinlichkeiten konvertieren
        out_cls = F.softmax(self.tm1(x), dim=1)  # (B, 36)
        out_type = F.softmax(self.tm2(x), dim=1)  # (B, 3)
        
        # Typ-Werte für jede Klasse zuordnen
        class_type_weights = out_type[:, self.class_type_map]  # (B, 36)
        s
        # Kombinierte Vorhersage (mit epsilon zur Vermeidung von Null)
        final_out = out_cls * (class_type_weights + 1e-8)
        return final_out, out_cls, out_type







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


