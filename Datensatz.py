import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, RandomAffine, Compose
from collections import defaultdict
import random
from PIL import Image
import matplotlib.pyplot as plt

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
    augment = Compose([
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=5),
        ToTensor()
    ])

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
        ascii_label = chr(class_list[label])  # Neues Mapping per Liste

        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(ascii_label)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
