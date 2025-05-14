import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_predictions(model, loader, device, class_names, n=160, page_size=16):
    model.eval()
    imgs, trues, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            p    = out.argmax(dim=1)
            imgs.append(x.cpu()); trues.append(y.cpu()); preds.append(p.cpu())
            if sum(t.size(0) for t in imgs) >= n:
                break

    imgs  = torch.cat(imgs)[:n]
    trues = torch.cat(trues)[:n]
    preds = torch.cat(preds)[:n]
    total = len(imgs)
    pages = (total + page_size - 1) // page_size

    for page in range(pages):
        start = page * page_size
        end   = min(start + page_size, total)
        batch_imgs  = imgs[start:end]
        batch_true  = trues[start:end]
        batch_pred  = preds[start:end]
        k = int(np.sqrt(len(batch_imgs)))
        fig, axes = plt.subplots(k, k, figsize=(k*2, k*2))
        for i, ax in enumerate(axes.flatten()):
            idx = start + i
            if idx < end:
                ax.imshow(batch_imgs[i].squeeze(), cmap='gray')
                ax.set_title(f"{class_names[batch_true[i]]} → {class_names[batch_pred[i]]}", fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        if page < pages-1:
            input(f"-- Strona {page+1}/{pages}. Enter aby kontynuować --")
