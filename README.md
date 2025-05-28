# Trening od zera z oversamplingiem, label smoothing, MixUp i Cosine LR
python main.py --csv_path data/fashion-mnist_train.csv --mode train --epochs 30 --batch_size 64 --lr_main 1e-3 --patience 5 --use_mixup --mixup_alpha 0.4

# Trening od zera + bezpieczny fine-tuning
python main.py --csv_path data/fashion-mnist_train.csv --mode train --epochs 30 --batch_size 64 --lr_main 1e-3 --patience 5 --use_mixup --mixup_alpha 0.4 --do_finetune --fine_tune_epochs 5 --lr_ft 1e-5 --ft_conf_thresh 0.9

# Sam fine-tuning (na backup_before_ft.pth)
python main.py --csv_path data/fashion-mnist_train.csv --mode train --resume_model backup_before_ft.pth --epochs 0 --batch_size 64 --do_finetune --fine_tune_epochs 5 --lr_ft 1e-5 --ft_conf_thresh 0.9

# Kontynuacja treningu od ostatniego checkpointu z obniżonym LR
python main.py --csv_path data/fashion-mnist_train.csv --mode train --resume_model model_final.pth --epochs 10 --batch_size 64 --lr_main 1e-4 --patience 3

# Ewaluacja końcowego modelu na split 20%
python main.py --csv_path data/fashion-mnist_train.csv --mode eval --resume_model model_final.pth

# Tylko test na zewnętrznym zbiorze _test.csv
python main.py --only_test --test_path data/fashion-mnist_test.csv --resume_model model_final.pth
