from Klassifikator import EarlyStopping
import torch
import torch.nn.functional as F

def klassifier_training(train_loader, test_loader, model, device, criterion, optimizer, scheduler, epochs=30):
    # EarlyStopping initialisieren (vor der Schleife!)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    best_val_acc = 0.0  # Tracking der besten Validation Accuracy
    best_epoch = 0      # Merken der besten Epoche

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 200 == 0:
                print(f"🔁 Epoch {epoch+1}, Step {i+1}/{len(train_loader)}: Batch Loss = {loss.item():.4f}")

        scheduler.step()

        # Validierung
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= len(test_loader)
        train_acc = 100.0 * correct_train / total_train
        val_acc = 100.0 * correct_val / total_val

        print(f"📊 Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, Val Loss = {val_loss:.4f}, LR = {scheduler.get_last_lr()}")

        # Early Stopping überprüfen
        early_stopping(val_loss, model)
        
        # Beste Accuracy aktualisieren
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        if early_stopping.early_stop:
            print(f"🛑 Early stopping bei Epoche {epoch+1} (Beste Val Acc: {best_val_acc:.2f}% in Epoche {best_epoch})")
            break

    # Nach dem Training: Bestes Modell wiederherstellen
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print("✅ Bestes Modell wurde wiederhergestellt")
    else:
        print("⚠️ Kein besseres Modell gefunden als Initialisierung")










def train_modular(model, train_loader, test_loader, device, optimizer, scheduler=None, alpha=0.5, epochs=50):
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_accuracy = 0.0

    def evaluate(model, loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels_cls, labels_type in loader:
                images = images.to(device)
                labels_cls = labels_cls.to(device)
                labels_type = labels_type.to(device)

                final_out, _, out_type = model(images)

                loss_cls = F.cross_entropy(final_out, labels_cls)
                loss_type = F.cross_entropy(out_type, labels_type)
                loss = alpha * loss_cls + (1 - alpha) * loss_type
                total_loss += loss.item()

                _, predicted = torch.max(final_out, 1)
                correct += (predicted == labels_cls).sum().item()
                total += labels_cls.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels_cls, labels_type) in enumerate(train_loader):
            images = images.to(device)
            labels_cls = labels_cls.to(device)
            labels_type = labels_type.to(device)

            optimizer.zero_grad()
            final_out, out_cls, out_type = model(images)

            loss_cls = F.cross_entropy(final_out, labels_cls)
            loss_type = F.cross_entropy(out_type, labels_type)
            loss = alpha * loss_cls + (1 - alpha) * loss_type

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        val_loss, val_acc = evaluate(model, test_loader)

        print(f"📊 Epoch {epoch+1}/{epochs}: Train Loss = {running_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%, LR = {scheduler.get_last_lr() if scheduler else 'N/A'}")

        # Bestes Modell speichern
        if val_acc > best_accuracy:
            best_accuracy = val_acc

        # Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"🛑 Early stopping in Epoche {epoch+1} (Beste Val Acc: {best_accuracy:.2f}%)")
            break

    # Bestes Modell laden
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print("✅ Bestes Modell wurde wiederhergestellt")
    else:
        print("⚠️ Kein besseres Modell gefunden als Initialisierung")

    return model




def train_type_classifier(train_loader, test_loader, model, device, criterion, optimizer, scheduler, epochs=30):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, _, y_type) in enumerate(train_loader):  # Achtung: y_type für Type-Klasse
            images, y_type = images.to(device), y_type.to(device)
            outputs = model(images)
            loss = criterion(outputs, y_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_type.size(0)
            correct_train += (predicted == y_type).sum().item()

            if (i + 1) % 200 == 0:
                print(f"🔁 Epoch {epoch+1}, Step {i+1}/{len(train_loader)}: Batch Loss = {loss.item():.4f}")

        scheduler.step()

        # Validierung
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        model.eval()
        with torch.no_grad():
            for images, _, y_type in test_loader:
                images, y_type = images.to(device), y_type.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, y_type).item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += y_type.size(0)
                correct_val += (predicted == y_type).sum().item()

        val_loss /= len(test_loader)
        train_acc = 100.0 * correct_train / total_train
        val_acc = 100.0 * correct_val / total_val

        print(f"📊 Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, Val Loss = {val_loss:.4f}, LR = {scheduler.get_last_lr()}")

        early_stopping(val_loss, model)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        if early_stopping.early_stop:
            print(f"🛑 Early stopping bei Epoche {epoch+1} (Beste Val Acc: {best_val_acc:.2f}% in Epoche {best_epoch})")
            break

    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print("✅ Bestes Modell wurde wiederhergestellt")
    else:
        print("⚠️ Kein besseres Modell gefunden als Initialisierung")
