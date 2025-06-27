import torch
from torchvision import datasets
from torchvision.transforms import Compose, RandomResizedCrop, ColorJitter, RandomAffine, ToTensor
from collections import defaultdict
import random
from PIL import Image
import matplotlib.pyplot as plt





def cutout(img, mask_size=8):
    """Cutout-Augenblick: Zufälliges Rechteck aus dem Bild entfernen."""
    h, w = img.shape[1], img.shape[2]
    top = random.randint(0, h - mask_size)
    left = random.randint(0, w - mask_size)

    img[:, top:top+mask_size, left:left+mask_size] = 0
    return img

def add_noise(img, noise_factor=0.2):
    """Fügt zufälliges Rauschen zum Bild hinzu."""
    img = img + noise_factor * torch.randn_like(img)
    img = torch.clamp(img, 0., 1.)  # Werte zwischen 0 und 1 begrenzen
    return img



def get_emnist_test_train():
    emnist = datasets.EMNIST(root='data', split='byclass', train=True, download=True)

    # Ziel-ASCII: 0–9, A–M, a–m → ergibt 36 Klassen
    class_list = list(range(48, 58)) + list(range(65, 78)) + list(range(97, 110))
    print("Ziel-ASCII:", class_list)
    print("Anzahl Zielklassen:", len(class_list))

    # Mappings: ASCII ⇄ Index
    class_to_index = {ascii_code: idx for idx, ascii_code in enumerate(class_list)}
    index_to_class = {v: k for k, v in class_to_index.items()}

    # EMNIST: Label → ASCII-Code
    mapping_dict = {i: code for i, code in enumerate(
        list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))
    )}
    
    # Nur relevante Labels extrahieren
    target_labels = [label for label, ascii_code in mapping_dict.items() if ascii_code in class_list]

    # Augmentierung (nur affine Transformationen – keine Spiegelung)
    # Augmentierungsstrategie
    augment = Compose([
    RandomResizedCrop(size=28, scale=(0.8, 1.0)),  # Zufällige Skalierung und Zuschneiden
    ColorJitter(brightness=0.2, contrast=0.2),      # Helligkeit und Kontrast zufällig ändern
    RandomAffine(
        degrees=0,                                  # Keine Rotation
        shear=5,                                    # Kleine Scherung
        translate=(0.1, 0.1),                       # Kleine Translationen
        scale=(0.9, 1.1)                            # Kleine Skalierungen
    ),
    ToTensor(),
    lambda x: add_noise(x),                          # Zufälliges Rauschen hinzufügen
    lambda x: cutout(x),                             # Cutout hinzufügen
    ] )

    # Zielanzahl pro Klasse
    samples_per_class_train = 5000
    samples_per_class_test = 1000
    total_needed = samples_per_class_train + samples_per_class_test

    # Alle Sample-Indices je Klasse sammeln
    class_samples = defaultdict(list)
    for idx in range(len(emnist)):
        _, label = emnist[idx]
        if label in target_labels:
            class_samples[label].append(idx)

    X_train, y_train, X_test, y_test = [], [], [], []

    for label in target_labels:
        indices = class_samples[label]
        num_real = len(indices)
        images = []

        ascii_code = mapping_dict[label]
        class_idx = class_to_index[ascii_code]

        if num_real >= total_needed:
            random.shuffle(indices)
            selected = indices[:total_needed]
            for idx in selected:
                img = Image.fromarray(emnist.data[idx].numpy(), mode='L')
                img_tensor = ToTensor()(img)
                images.append(img_tensor)
        else:
            print(f"⚠️ Klasse {chr(ascii_code)}: nur {num_real} echte Bilder – augmentiere {total_needed - num_real} zusätzlich.")
            for idx in indices:
                img = Image.fromarray(emnist.data[idx].numpy(), mode='L')
                img_tensor = ToTensor()(img)
                images.append(img_tensor)
            for i in range(total_needed - num_real):
                idx = indices[i % num_real]
                img = Image.fromarray(emnist.data[idx].numpy(), mode='L')
                img_aug = augment(img)
                images.append(img_aug)

        # Label zuweisen und mischen
        combined = list(zip(images, [torch.tensor(class_idx)] * total_needed))
        random.shuffle(combined)
        images, labels = zip(*combined)

        X_train.extend(images[:samples_per_class_train])
        y_train.extend(labels[:samples_per_class_train])
        X_test.extend(images[samples_per_class_train:])
        y_test.extend(labels[samples_per_class_train:])

    # Shuffle der gesamten Daten
    train_combined = list(zip(X_train, y_train))
    test_combined = list(zip(X_test, y_test))
    random.shuffle(train_combined)
    random.shuffle(test_combined)
    X_train, y_train = zip(*train_combined)
    X_test, y_test = zip(*test_combined)

    # In Tensorstapel umwandeln
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    print("✅ Trainingsdaten:", X_train.shape, y_train.shape)
    print("✅ Testdaten:", X_test.shape, y_test.shape)
    class_list = list('0123456789ABCDEFGHIJKLMabcdefghijklm')

    return X_train, y_train, X_test, y_test, class_list









def show_random_samples(X, y, class_list, n=10):
    """
    Zeigt n zufällige Bilder aus einem Tensor-Datensatz mit zugehörigen ASCII-Labels.

    Args:
        X (Tensor): Bilddaten (N, 1, 28, 28)
        y (Tensor): Labels (N,)
        class_list (list): Liste von ASCII-Zeichen, geordnet wie im Training (Index = Label)
        n (int): Anzahl der Bilder
    """
    assert len(X) == len(y), "X und y müssen gleich lang sein."
    indices = random.sample(range(len(X)), n)

    plt.figure(figsize=(15, 2))
    for i, idx in enumerate(indices):
        img = X[idx].squeeze().numpy()
        label = y[idx].item()
        ascii_label = str(class_list[label])  # Neues Mapping per Liste

        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(ascii_label)
        plt.axis('off')
    plt.tight_layout()
    plt.show()





def create_class_type_map(class_list):
    class_type_map = []
    for c in class_list:
        if c.isdigit():
            class_type_map.append(0)
        elif c.isupper():
            class_type_map.append(1)
        else:
            class_type_map.append(2)
    return class_type_map

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