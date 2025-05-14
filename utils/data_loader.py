import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def make_dataloaders(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    X = df.iloc[:,1:].values.astype('float32') / 255.0
    y = df.iloc[:,0].values.astype('int64')
    X = X.reshape(-1,1,28,28)

    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr))
    test_ds  = TensorDataset(torch.tensor(x_te), torch.tensor(y_te))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)
    return train_loader, test_loader, train_ds
