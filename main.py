import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from models.cnn import CNN
from train import train_epoch, eval_epoch
from utils.data_loader import make_dataloaders
from utils.early_stopping import EarlyStopping
from finetune import fine_tune
from evaluate import evaluate_model
from utils.visualize import visualize_predictions

def ensure_dirs():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path',      required=True,
                   help="ścieżka do CSV z danymi")
    p.add_argument('--resume_model',  default=None,
                   help="plik .pth do wznowienia lub ewaluacji")
    p.add_argument('--mode', choices=['train','eval'], default='train')
    p.add_argument('--epochs',           type=int, default=5)
    p.add_argument('--batch_size',       type=int, default=64)
    p.add_argument('--lr_main',          type=float, default=1e-3)
    p.add_argument('--patience',         type=int, default=3)
    p.add_argument('--do_finetune',      action='store_true',
                   help="włącz bezpieczny fine-tuning")
    p.add_argument('--fine_tune_epochs', type=int, default=3)
    p.add_argument('--lr_ft',            type=float, default=1e-5)
    p.add_argument('--ft_conf_thresh',   type=float, default=0.9,
                   help="próg pewności dla FT [0–1]")
    p.add_argument('--confusion',        action='store_true',
                   help="pokaż macierz pomyłek")
    p.add_argument('--save_name',        default='model_final.pth')
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = [
        'T-shirt/top','Trouser','Pullover','Dress','Coat',
        'Sandal','Shirt','Sneaker','Bag','Ankle boot'
    ]

    train_loader, test_loader, train_ds = make_dataloaders(
        args.csv_path, args.batch_size
    )
    model = CNN().to(device)

    if args.resume_model:
        ckpt = torch.load(args.resume_model, map_location=device)
        model.load_state_dict(ckpt.get('model_state', ckpt))

    # TRAIN
    if args.mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_main)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=1, factor=0.5
        )
        es = EarlyStopping(patience=args.patience)
        for ep in range(1, args.epochs+1):
            tr_loss = train_epoch(train_loader, model, optimizer,
                                  nn.CrossEntropyLoss(), device)
            val_loss, val_acc = eval_epoch(test_loader, model,
                                           nn.CrossEntropyLoss(), device)
            scheduler.step(val_loss)
            print(f"Ep {ep}/{args.epochs}  TrL:{tr_loss:.4f}  ValL:{val_loss:.4f}  ValA:{val_acc:.4f}")
            if es(val_loss):
                print("EarlyStopping.")
                break
        torch.save({'model_state': model.state_dict()}, 'backup_before_ft.pth')

    # OPTIONAL FINE-TUNE
    if args.mode == 'train' and args.do_finetune:
        fine_tune(model, train_ds, test_loader, device,
                  epochs=args.fine_tune_epochs,
                  lr=args.lr_ft,
                  patience=args.patience,
                  batch_size=args.batch_size,
                  conf_thresh=args.ft_conf_thresh)

    # EVAL + SAVE + PLOTS
    if args.mode in ('train','eval'):
        torch.save({'model_state': model.state_dict()}, args.save_name)
        print(f"Saved model to {args.save_name}")
        evaluate_model(model, test_loader, device,
                       class_names, show_confusion=args.confusion)
        visualize_predictions(model, test_loader, device, class_names)

if __name__ == '__main__':
    main()
