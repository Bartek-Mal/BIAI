import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import eval_epoch

def plot_confusion_matrix(model, loader, device, class_names):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds.extend(out.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.show()

def evaluate_model(model, test_loader, device, class_names, show_confusion=False):
    _, acc = eval_epoch(test_loader, model, None, device)
    print(f"Test accuracy: {acc*100:.2f}%")
    if show_confusion:
        plot_confusion_matrix(model, test_loader, device, class_names)
