# BIAI
# 1) Trening od zera (z zapisem backup_before_ft.pth)
python main.py --csv_path data/fashion-mnist_train.csv --mode train --epochs 10 --batch_size 64 --lr_main 1e-3 --patience 3

# 2) Trening od zera + bezpieczny fine-tuning
python main.py --csv_path data/fashion-mnist_train.csv --mode train --epochs 10 --batch_size 64 --lr_main 1e-3 --patience 3 --do_finetune --fine_tune_epochs 3 --lr_ft 1e-5 --ft_conf_thresh 0.9

# 3) Sam fine-tuning (tylko na błędach, pomijamy główny trening)
python main.py --csv_path data/fashion-mnist_train.csv --mode train --resume_model backup_before_ft.pth --epochs 0 --batch_size 64 --do_finetune --fine_tune_epochs 3 --lr_ft 1e-5 --ft_conf_thresh 0.9

# 4) Kontynuacja treningu od ostatniego checkpointu
python main.py --csv_path data/fashion-mnist_train.csv --mode train --resume_model model_final.pth --epochs 5 --batch_size 64 --lr_main 1e-4 --patience 2

# 5a) Ewaluacja końcowego modelu (bez macierzy)
python main.py --csv_path data/fashion-mnist_train.csv --mode eval --resume_model model_final.pth

# 5b) Ewaluacja + macierz pomyłek
python main.py --csv_path data/fashion-mnist_train.csv --mode eval --resume_model model_final.pth --confusion
