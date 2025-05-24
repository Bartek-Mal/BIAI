# 1) Trening od zera z oversamplingiem, label smoothing, MixUp i Cosine LR
python main.py --csv_path data/fashion-mnist_train.csv --mode train --epochs 30 --batch_size 64 --lr_main 1e-3 --patience 5 --use_mixup --mixup_alpha 0.4

# 2) Trening od zera + bezpieczny fine-tuning (douczanie na pewnych błędach)
python main.py --csv_path data/fashion-mnist_train.csv --mode train --epochs 30 --batch_size 64 --lr_main 1e-3 --patience 5 --use_mixup --mixup_alpha 0.4 --do_finetune --fine_tune_epochs 5 --lr_ft 1e-5 --ft_conf_thresh 0.9

# 3) Sam fine-tuning (pomijamy główny trening, douczamy z backup_before_ft.pth)
python main.py --csv_path data/fashion-mnist_train.csv --mode train --resume_model backup_before_ft.pth --epochs 0 --batch_size 64 --do_finetune --fine_tune_epochs 5 --lr_ft 1e-5 --ft_conf_thresh 0.9

# 4) Kontynuacja treningu od ostatniego checkpointu z obniżonym LR
python main.py --csv_path data/fashion-mnist_train.csv --mode train --resume_model model_final.pth --epochs 10 --batch_size 64 --lr_main 1e-4 --patience 3

# 5a) Ewaluacja końcowego modelu bez macierzy pomyłek
python main.py --csv_path data/fashion-mnist_train.csv --mode eval --resume_model model_final.pth

# 5b) Ewaluacja końcowego modelu + wyświetlenie macierzy pomyłek
python main.py --csv_path data/fashion-mnist_train.csv --mode eval --resume_model model_final.pth --confusion
