import torch
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, device, class_list):
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Keine Gradientenberechnung, da wir nur evaluieren
        n_correct = 0
        n_samples = 0
        n_class_correct = defaultdict(int)
        n_class_samples = defaultdict(int)

        # Iteriere über den Testdatensatz
        for images, labels in test_loader:
            images = images.to(device)  # Bilder auf das gleiche Gerät verschieben (GPU oder CPU)
            labels = labels.to(device)  # Labels auf das gleiche Gerät verschieben (GPU oder CPU)
            
            # Vorwärtsdurchlauf
            outputs = model(images)
            
            # Vorhersagen
            _, predicted = torch.max(outputs, 1)

            # Update der Gesamtmetriken
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            # Update der Metriken pro Klasse
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

            # Speichere alle Labels und Vorhersagen für die Berechnung der weiteren Metriken
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Berechnung der Gesamtgenauigkeit
        acc = 100.0 * n_correct / n_samples
        print(f'Gesamtgenauigkeit des Netzwerks: {acc:.2f} %')

        # Berechnung der Klasse-genauen Genauigkeit
        for label in sorted(n_class_samples.keys()):
            ascii_char = class_list[label]
            class_acc = 100.0 * n_class_correct[label] / n_class_samples[label]
            print(f'Genauigkeit für Klasse {ascii_char}: {class_acc:.2f} %')

        # Berechnung der Precision, Recall und F1-Score für jede Klasse
        precision = precision_score(all_labels, all_predictions, average=None, labels=np.unique(all_labels))
        recall = recall_score(all_labels, all_predictions, average=None, labels=np.unique(all_labels))
        f1 = f1_score(all_labels, all_predictions, average=None, labels=np.unique(all_labels))
        
        # Berechne den durchschnittlichen F1-Score
        avg_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print("\nPrecision, Recall, F1-Score pro Klasse:")
        for i, ascii_char in enumerate(class_list):
            print(f"Klasse {ascii_char}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-Score={f1[i]:.2f}")
        
        print(f"\nDurchschnittlicher F1-Score (gewichtet): {avg_f1:.2f}")

        # Berechnung der Konfusionsmatrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\nKonfusionsmatrix:\n{cm}")

        # Berechnung der ROC-AUC (für Multiklassen kann man dies auch für jedes Label einzeln berechnen)
        try:
            roc_auc = roc_auc_score(all_labels, model_output_to_probs(model, test_loader, device), multi_class='ovr', average='weighted')
            print(f"\nDurchschnittliche ROC-AUC: {roc_auc:.2f}")
        except ValueError:
            print("\nROC-AUC konnte nicht berechnet werden (möglicherweise nicht geeignet für das Problem).")

        return acc

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