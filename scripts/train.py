from tqdm import tqdm
import torch
import time
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

def train(model, device, train_loader, val_loader, test_loader, criterion, optimizer, epochs=10):
    model.to(device)
    os.makedirs("results", exist_ok=True)

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, target in train_loader:
            data = torch.nn.functional.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * data.size(0)
            pred = outputs.argmax(dim=1)
            correct_train += pred.eq(target).sum().item()
            total_train += target.size(0)

        avg_train_loss = running_train_loss / total_train
        train_accuracy = 100. * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = torch.nn.functional.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct_val += pred.eq(target).sum().item()
                total_val += target.size(0)

        avg_val_loss = val_loss / total_val
        val_accuracy = 100. * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # --- Final Test Eval (optional every epoch) ---
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = torch.nn.functional.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct_test += pred.eq(target).sum().item()
                total_test += target.size(0)

        avg_test_loss = test_loss / total_test
        test_accuracy = 100. * correct_test / total_test
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        # --- Print Epoch Summary ---
        tqdm.write(f"Epoch {epoch+1}/{epochs} | "
                   f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
                   f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}% | "
                   f"Test Acc: {test_accuracy:.2f}%")

    # --- Plot Loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_curve.png")
    print("📊 Saved: results/loss_curve.png")

    # --- Plot Accuracy ---
    plt.figure(figsize=(8, 4))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/accuracy_curve.png")
    print("📊 Saved: results/accuracy_curve.png")

    print(f"\n📊 Final Test Accuracy: {test_accuracies[-1]:.2f}%")
