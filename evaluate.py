import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os

from scripts.dataloader import get_loaders
from scripts.AlexNet import AlexNet


def evaluate(model, device, loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    all_losses = []

    with torch.no_grad():
        for data, target in loader:
            # KEEPING interpolate to resize 32x32 → 224x224
            data = torch.nn.functional.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            all_losses.extend(nn.functional.cross_entropy(output, target, reduction='none').cpu().numpy())
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_losses)


def plot_loss_distribution(losses, tag):
    plt.figure(figsize=(8, 4))
    plt.title(f"Prediction Loss per Sample ({tag})")
    sns.histplot(losses, bins=50, kde=True, color='crimson')
    plt.xlabel("Cross Entropy Loss")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(f"results/loss_distribution_{tag}.png")
    print(f"📊 Saved: results/loss_distribution_{tag}.png")


def plot_confusion_matrix(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix ({tag})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{tag}.png")
    print(f"📊 Saved: results/confusion_matrix_{tag}.png")


def plot_accuracy_per_class(y_true, y_pred, tag, num_classes=10):
    correct = [0] * num_classes
    total = [0] * num_classes

    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1

    acc_per_class = [100 * c / t if t != 0 else 0 for c, t in zip(correct, total)]

    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    plt.figure(figsize=(10, 5))
    bars = sns.barplot(x=class_names, y=acc_per_class, palette='viridis')
    plt.title(f"Accuracy per Class ({tag})")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)

    for bar, acc in zip(bars.patches, acc_per_class):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.2f}%",
            ha='center',
            va='bottom',
            fontsize=8,
            color='black'
        )

    plt.tight_layout()
    plt.savefig(f"results/accuracy_per_class_{tag}.png")
    print(f"📊 Saved: results/accuracy_per_class_{tag}.png")


def plot_pred_vs_true(y_true, y_pred, tag):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    plt.plot([0, 9], [0, 9], 'r--')
    plt.title(f"Predicted vs True Labels ({tag})")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"results/pred_vs_true_{tag}.png")
    print(f"📊 Saved: results/pred_vs_true_{tag}.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    # Load all datasets
    train_loader, val_loader, test_loader = get_loaders(batch_size=512)

    # Load model
    model = AlexNet(num_classes=10)
    model.load_state_dict(torch.load("wts/alexnet_model.pth", map_location=device))
    model = model.to(device)
    print("✅ Model loaded")

    criterion = nn.CrossEntropyLoss()

    for tag, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
        print(f"\n📈 Evaluating on {tag.upper()} set...")
        y_pred, y_true, losses = evaluate(model, device, loader, criterion)

        final_acc = accuracy_score(y_true, y_pred) * 100
        print(f"✅ Accuracy on {tag} set: {final_acc:.2f}%")

        # Plots
        plot_loss_distribution(losses, tag)
        plot_confusion_matrix(y_true, y_pred, tag)
        plot_accuracy_per_class(y_true, y_pred, tag)
        plot_pred_vs_true(y_true, y_pred, tag)

    print("\n✅ All evaluations complete. Results saved in 'results/' folder.")
