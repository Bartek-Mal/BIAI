import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def make_ft_loader(wrong_imgs, wrong_lbls, base_ds,
                   batch_size, frac_wrong=0.5, conf_thresh=0.9, device='cpu'):
    # wybieramy tylko te błędy, które model sklasyfikował z pewnością >= conf_thresh
    model = base_ds.dataset.model  # upewnij się, że dataset przechowuje referencję do modelu
    model.eval()
    keep_imgs, keep_lbls = [], []
    with torch.no_grad():
        for img, lbl in zip(wrong_imgs, wrong_lbls):
            inp = img.unsqueeze(0).to(device)
            probs = F.softmax(model(inp), dim=1)
            conf, _ = torch.max(probs, dim=1)
            if conf.item() >= conf_thresh:
                keep_imgs.append(img)
                keep_lbls.append(lbl)

    n_keep = len(keep_imgs)
    if n_keep == 0:
        return None

    n_rand = int(n_keep * (1 - frac_wrong) / frac_wrong)
    idxs = np.random.choice(len(base_ds), n_rand, replace=False)
    rand_imgs = torch.stack([base_ds[i][0] for i in idxs])
    rand_lbls = torch.tensor([base_ds[i][1] for i in idxs])

    imgs = torch.cat([torch.stack(keep_imgs), rand_imgs])
    lbls = torch.cat([torch.tensor(keep_lbls), rand_lbls])
    ds   = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
