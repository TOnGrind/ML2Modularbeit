import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import optuna
import numpy as np
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor, Compose, RandomAffine
from torch.utils.data import Subset, ConcatDataset
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import copy
from Datensatz import get_emnist_test_train
import optuna
from Klassifikator import get_objective, ResNet18, EarlyStopping


# Device-Konfiguration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Verwende Gerät: {device}")

# 1. Datensatzvorbereitung
# ========================
X_train, y_train, X_test, y_test, class_list  = get_emnist_test_train()


# 2. Modelldefinitionen
# =====================

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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
        
        # Kombinierte Vorhersage (mit epsilon zur Vermeidung von Null)
        final_out = out_cls * (class_type_weights + 1e-8)
        return final_out, out_cls, out_type

# 3. Hilfsklassen und -funktionen
# ===============================

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0
        return self.early_stop

def create_type_labels(labels, class_list):
    type_labels = []
    for label in labels:
        char = class_list[int(label)]
        if char.isdigit():
            type_labels.append(0)
        elif char.isupper():
            type_labels.append(1)
        else:
            type_labels.append(2)
    return torch.tensor(type_labels, dtype=torch.long)

# 4. Trainings- und Evaluationsfunktionen
# =======================================

def train_evaluate_simple(model, train_loader, test_loader, device, n_epochs=20):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    
    for epoch in range(n_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_resnet.pth")
        
        print(f"Epoch {epoch+1}/{n_epochs}, Accuracy: {acc:.4f}")
    
    return best_acc

def train_evaluate_modular(trial, train_loader, test_loader, class_type_map, device):
    # Hyperparameter-Vorschläge
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    alpha = trial.suggest_float('alpha', 0.5, 0.9)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # Modelle initialisieren
    tm1 = ResNet18(num_classes=len(class_list)).to(device)
    tm1.load_state_dict(torch.load("best_resnet.pth", map_location=device))
    
    # Nur letzte Schicht von TM1 trainierbar machen
    for param in tm1.parameters():
        param.requires_grad = False
    for param in tm1.linear.parameters():
        param.requires_grad = True
    
    tm2 = TypeClassifier(dropout_rate=dropout).to(device)
    model = ModularClassifier(tm1, tm2, class_type_map).to(device)
    
    # Optimierer und Loss
    optimizer = optim.Adam([
        {'params': tm2.parameters()},
        {'params': tm1.linear.parameters()}
    ], lr=lr)
    
    early_stopping = EarlyStopping(patience=5)
    
    # Trainingsschleife
    best_accuracy = 0
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        
        for images, labels_cls, labels_type in train_loader:
            images = images.to(device)
            labels_cls = labels_cls.to(device)
            labels_type = labels_type.to(device)
            
            optimizer.zero_grad()
            final_out, out_cls, out_type = model(images)
            
            # Kombinierter Loss
            loss_cls = F.cross_entropy(final_out, labels_cls)
            loss_type = F.cross_entropy(out_type, labels_type)
            loss = alpha * loss_cls + (1 - alpha) * loss_type
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels_cls, _ in test_loader:
                images = images.to(device)
                labels_cls = labels_cls.to(device)
                
                final_out, _, _ = model(images)
                _, predicted = torch.max(final_out, 1)
                
                correct += (predicted == labels_cls).sum().item()
                total += labels_cls.size(0)
        
        accuracy = correct / total
        trial.report(accuracy, epoch)
        
        # Beste Genauigkeit speichern
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        # Early Stopping
        if early_stopping(accuracy):
            break
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_accuracy

# 5. Hauptprogramm
# ================

def main():
    # Daten laden
    X_train, y_train, X_test, y_test, class_list = get_emnist_test_train()
    
    # Klassen-Typ-Mapping erstellen
    class_type_map = []
    for c in class_list:
        if c.isdigit():
            class_type_map.append(0)
        elif c.isupper():
            class_type_map.append(1)
        else:
            class_type_map.append(2)
    
    # Typ-Labels erstellen
    y_train_type = create_type_labels(y_train, class_list)
    y_test_type = create_type_labels(y_test, class_list)
    
    # Datensätze erstellen
    train_dataset = TensorDataset(X_train, y_train, y_train_type)
    test_dataset = TensorDataset(X_test, y_test, y_test_type)
    
    # 1. Schritt: Einfachen Klassifikator trainieren
    print("\n" + "="*50)
    print("Training des einfachen Klassifikators")
    print("="*50)
    
    # Für schnelleres Testen: Kleinere Datensätze verwenden
    # train_dataset = TensorDataset(X_train[:1000], y_train[:1000], y_train_type[:1000])
    # test_dataset = TensorDataset(X_test[:200], y_test[:200], y_test_type[:200])
    
    simple_model = ResNet18(num_classes=len(class_list)).to(device)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Trainiere einfachen Klassifikator
    simple_acc = train_evaluate_simple(
        simple_model, 
        DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True),
        DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False),
        device,
        n_epochs=10
    )
    print(f"🚀 Beste Genauigkeit des einfachen Systems: {simple_acc:.4f}")
    
    # 2. Schritt: Modulares System trainieren
    print("\n" + "="*50)
    print("Training des modularen Klassifikators")
    print("="*50)
    
    
    """
    # Optuna-Studie für modulares System
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: train_evaluate_modular(
            trial, 
            DataLoader(train_dataset, batch_size=128, shuffle=True),
            DataLoader(test_dataset, batch_size=128, shuffle=False),
            class_type_map,
            device
        ), 
        n_trials=5
    )
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(get_objective(
          train_dataset=train_dataset,
          test_dataset=test_dataset,
          device=device,
          model=ResNet18(num_classes=len(class_list)).to(device),
          early_stopping=EarlyStopping(patience=4)), n_trials=20)

    # Beste Parameter anzeigen
    print("🎯 Beste Hyperparameter:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
        
    # Beste Hyperparameter ausgeben
    print("\n🎯 Beste Hyperparameter für modulares System:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    # Finales Modell mit besten Hyperparametern trainieren
    best_params = study.best_params
    
    tm1 = ResNet18(num_classes=len(class_list)).to(device)
    tm1.load_state_dict(torch.load("best_resnet.pth", map_location=device))
    
    for param in tm1.parameters():
        param.requires_grad = False
    for param in tm1.linear.parameters():
        param.requires_grad = True
    
    tm2 = TypeClassifier(dropout_rate=best_params['dropout']).to(device)
    model = ModularClassifier(tm1, tm2, class_type_map).to(device)
    
    optimizer = optim.Adam([
        {'params': tm2.parameters()},
        {'params': tm1.linear.parameters()}
    ], lr=best_params['lr'])
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # Training mit Early Stopping
    early_stopping = EarlyStopping(patience=5)
    best_accuracy = 0
    
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        
        for images, labels_cls, labels_type in train_loader:
            images = images.to(device)
            labels_cls = labels_cls.to(device)
            labels_type = labels_type.to(device)
            
            optimizer.zero_grad()
            final_out, out_cls, out_type = model(images)
            
            # Kombinierter Loss
            loss_cls = F.cross_entropy(final_out, labels_cls)
            loss_type = F.cross_entropy(out_type, labels_type)
            loss = best_params['alpha'] * loss_cls + (1 - best_params['alpha']) * loss_type
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels_cls, _ in test_loader:
                images = images.to(device)
                labels_cls = labels_cls.to(device)
                
                final_out, _, _ = model(images)
                _, predicted = torch.max(final_out, 1)
                
                correct += (predicted == labels_cls).sum().item()
                total += labels_cls.size(0)
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Early Stopping und Model speichern
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_modular_model.pth")
        
        if early_stopping(accuracy):
            print("Early stopping triggered")
            break
    
    print(f"🚀 Beste Genauigkeit des modularen Systems: {best_accuracy:.4f}")
    
    # 3. Schritt: Vergleich der Modelle
    print("\n" + "="*50)
    print("Vergleich der Modelle")
    print("="*50)
    
    # Lade beste Modelle
    simple_model.load_state_dict(torch.load("best_resnet.pth", map_location=device))
    model.load_state_dict(torch.load("best_modular_model.pth", map_location=device))
    
    simple_model.eval()
    model.eval()
    
    simple_correct = 0
    modular_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels_cls, _ in test_loader:
            images = images.to(device)
            labels_cls = labels_cls.to(device)
            
            # Einfaches Modell
            outputs_simple = simple_model(images)
            _, predicted_simple = torch.max(outputs_simple, 1)
            simple_correct += (predicted_simple == labels_cls).sum().item()
            
            # Modulares Modell
            final_out, _, _ = model(images)
            _, predicted_modular = torch.max(final_out, 1)
            modular_correct += (predicted_modular == labels_cls).sum().item()
            
            total += labels_cls.size(0)
    
    simple_acc = simple_correct / total
    modular_acc = modular_correct / total
    
    print("\n🔥 Vergleich der Modelle:")
    print(f"Einfacher Klassifikator: {simple_acc:.4f}")
    print(f"Modularer Klassifikator: {modular_acc:.4f}")
    print(f"Verbesserung: {modular_acc - simple_acc:+.4f}")
    
    # Genauigkeit nach Typen aufschlüsseln
    digit_indices = [i for i, c in enumerate(class_list) if c.isdigit()]
    upper_indices = [i for i, c in enumerate(class_list) if c.isupper()]
    lower_indices = [i for i, c in enumerate(class_list) if c.islower()]
    
    def calculate_accuracy(indices, model, is_modular=False):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels_cls, _ in test_loader:
                images = images.to(device)
                labels_cls = labels_cls.to(device)
                
                # Nur relevante Indizes betrachten
                mask = torch.isin(labels_cls, torch.tensor(indices).to(device))
                if not mask.any():
                    continue
                
                images = images[mask]
                labels_cls = labels_cls[mask]
                
                if is_modular:
                    final_out, _, _ = model(images)
                    outputs = final_out
                else:
                    outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels_cls).sum().item()
                total += labels_cls.size(0)
        
        return correct / total if total > 0 else 0
    
    print("\n🔍 Genauigkeit nach Typen:")
    print(f"Ziffern (einfach): {calculate_accuracy(digit_indices, simple_model):.4f}")
    print(f"Ziffern (modular): {calculate_accuracy(digit_indices, model, True):.4f}")
    print(f"Großbuchstaben (einfach): {calculate_accuracy(upper_indices, simple_model):.4f}")
    print(f"Großbuchstaben (modular): {calculate_accuracy(upper_indices, model, True):.4f}")
    print(f"Kleinbuchstaben (einfach): {calculate_accuracy(lower_indices, simple_model):.4f}")
    print(f"Kleinbuchstaben (modular): {calculate_accuracy(lower_indices, model, True):.4f}")

if __name__ == "__main__":
    main()