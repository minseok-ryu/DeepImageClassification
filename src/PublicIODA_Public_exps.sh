python main.py --lr 0.003 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 32 --net_archit resnet50 --weight_decay 0.0005 --phase train
python main.py --lr 0.001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 128 --net_archit resnet101 --weight_decay 0.0 --phase train
python main.py --lr 0.001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 16 --net_archit resnet101 --weight_decay 1e-05 --phase train --apply_grayScale --augmnt
python main.py --lr 0.0001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 64 --net_archit resnet50 --weight_decay 5e-05 --phase train --augmnt
python main.py --lr 0.003 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 128 --net_archit resnet50 --weight_decay 0.0 --phase train --apply_CLAHE
python main.py --lr 0.001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 32 --net_archit resnet101 --weight_decay 0.0005 --phase train --apply_CLAHE --apply_gaussianBlur --augmnt
python main.py --lr 0.001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 16 --net_archit resnet50 --weight_decay 5e-05 --phase train --apply_CLAHE
python main.py --lr 0.0001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 128 --net_archit resnet50 --weight_decay 0.0005 --phase train --apply_grayScale
python main.py --lr 0.0001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 16 --net_archit resnet50 --weight_decay 1e-05 --phase train --apply_grayScale --augmnt
python main.py --lr 0.003 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 64 --net_archit resnet101 --weight_decay 1e-05 --phase train --apply_grayScale
python main.py --lr 0.003 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 16 --net_archit resnet101 --weight_decay 0.0005 --phase train --apply_CLAHE
python main.py --lr 0.003 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 128 --net_archit resnet101 --weight_decay 5e-05 --phase train
python main.py --lr 0.0001 --num_epochs 2000 --early_stopping 500 --image_resize 128 --bs_train 16 --net_archit resnet50 --weight_decay 0.0001 --phase train --augmnt
