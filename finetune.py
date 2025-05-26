import torch
import torch.nn as nn
import torch.optim as optim
from train import train_epoch, eval_epoch
from utils.early_stopping import EarlyStopping
from utils.fine_tune_loader import make_ft_loader

def fine_tune(model, train_ds, test_loader, device,
              epochs, lr, patience, batch_size, conf_thresh):
    # backup
    torch.save(model.state_dict(), 'backup_before_ft.pth')

    # zbieramy błędne obrazy
    model.eval()
    wrong_imgs, wrong_lbls = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out   = model(x)
            preds = out.argmax(dim=1)
            mask  = preds != y
            if mask.any():
                wrong_imgs.append(x[mask].cpu())
                wrong_lbls.append(y[mask].cpu())

    if not wrong_imgs:
        print("Brak błędów — pomijam fine-tuning.")
        return

    wrong_imgs = torch.cat(wrong_imgs)
    wrong_lbls = torch.cat(wrong_lbls)
    ft_loader = make_ft_loader(
        wrong_imgs, wrong_lbls, train_ds,
        model=model,
        batch_size=batch_size, frac_wrong=0.5,
        conf_thresh=conf_thresh, device=device
    )
    if ft_loader is None:
        print("Żadne błędy nie spełniają progu pewności — skip FT.")
        return
    # mierzenie przed FT
    _, acc_before = eval_epoch(test_loader, model, None, device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=1, factor=0.5
    )
    es = EarlyStopping(patience=patience)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        ft_loss = train_epoch(ft_loader, model, optimizer, criterion, device)
        _, acc = eval_epoch(test_loader, model, None, device)
        scheduler.step(ft_loss)
        print(f"[FT] Ep{ep}/{epochs}  FT_loss:{ft_loss:.4f}  Test_acc:{acc:.4f}")
        if es(ft_loss):
            print("EarlyStopping FT.")
            break

    # restore jeśli FT pogorszył
    _, acc_after  = eval_epoch(test_loader, model, None, device)
    
    if acc_after < acc_before:
        model.load_state_dict(torch.load('backup_before_ft.pth'))
        print(f"FT pogorszył wynik ({acc_after:.4f} < {acc_before:.4f}); przywrócono oryginał.")
    else:
        print(f"FT poprawił wynik ({acc_after:.4f} ≥ {acc_before:.4f}); zostawiam nowy model.")