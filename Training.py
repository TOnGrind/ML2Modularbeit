from Klassifikator import EarlyStopping
import torch


def _1_2training(train_loader,test_loader,model,device,criterion,optimizer,scheduler,epochs = 30):
    early_stopping = EarlyStopping()
    for epoch in range(epoch):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

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
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping ausgelöst bei Epoch {epoch+1} (Val Loss: {val_loss:.4f})")
            break